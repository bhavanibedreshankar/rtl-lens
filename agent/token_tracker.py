"""Token usage tracking and budget enforcement.

Registered as a LangChain callback on every model call made anywhere in the
LangGraph run. Records prompt/completion tokens per graph node (LangGraph tags
each model invocation with `metadata["langgraph_node"]`), persists a running
total, and raises once the configured budget is exceeded so a runaway
investigation loop can't silently burn through the user's API spend.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from agent.logging_config import get_logger, log_event
from config.schema import BudgetConfig

logger = get_logger("token_tracker")

# Rough $/1M tokens, blended estimate for cost reporting only (not billing-accurate).
_COST_PER_MILLION = {
    "default_prompt": 3.00,
    "default_completion": 15.00,
}


class TokenBudgetExceeded(RuntimeError):
    def __init__(self, spent: int, budget: int):
        super().__init__(f"Token budget exceeded: {spent} > {budget}")
        self.spent = spent
        self.budget = budget


@dataclass
class NodeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenTracker(BaseCallbackHandler):
    budget: BudgetConfig
    per_node: dict[str, NodeUsage] = field(default_factory=lambda: defaultdict(NodeUsage))
    _warned: bool = False
    # run_id -> node name, recorded at call start: on_llm_end's own kwargs don't
    # reliably carry `metadata` back (LangChain callback versions vary on this),
    # but on_chat_model_start always does, so that's where the mapping is captured.
    _run_node: dict[Any, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(u.total for u in self.per_node.values())

    def estimated_cost_usd(self) -> float:
        prompt = sum(u.prompt_tokens for u in self.per_node.values())
        completion = sum(u.completion_tokens for u in self.per_node.values())
        return (
            prompt / 1_000_000 * _COST_PER_MILLION["default_prompt"]
            + completion / 1_000_000 * _COST_PER_MILLION["default_completion"]
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kwargs) -> None:
        self._run_node[run_id] = (metadata or {}).get("langgraph_node", "unknown")

    def on_llm_end(self, response: Any, *, run_id, parent_run_id=None, **kwargs) -> None:
        node = self._run_node.pop(run_id, None) or (kwargs.get("metadata") or {}).get("langgraph_node", "unknown")
        prompt_tokens, completion_tokens = _extract_usage(response)
        usage = self.per_node[node]
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.calls += 1

        log_event(
            logger,
            "llm_call",
            node=node,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            running_total=self.total_tokens,
        )
        self._check_budget()

    def _check_budget(self) -> None:
        pct = 100 * self.total_tokens / self.budget.max_tokens_per_run
        if pct >= self.budget.warn_at_pct and not self._warned:
            self._warned = True
            log_event(
                logger,
                f"Token usage at {pct:.0f}% of budget",
                level=30,  # WARNING
                total_tokens=self.total_tokens,
                budget=self.budget.max_tokens_per_run,
            )
        if self.total_tokens > self.budget.max_tokens_per_run:
            raise TokenBudgetExceeded(self.total_tokens, self.budget.max_tokens_per_run)

    def as_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
            "budget": self.budget.max_tokens_per_run,
            "per_node": {
                node: {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total,
                    "calls": u.calls,
                }
                for node, u in self.per_node.items()
            },
        }

    def write_report(self, run_dir: Path) -> None:
        """Merges into any existing report rather than overwriting it: a full
        run spans multiple CLI invocations (`debug`, `more-evidence`,
        `approve`/`reject`), each with its own fresh tracker — without merging,
        every phase's write would clobber the previous phases' token counts
        with whatever this phase alone spent (often zero, e.g. apply_and_verify
        makes no LLM calls at all)."""
        path = run_dir / "token_report.json"
        merged = json.loads(path.read_text()) if path.is_file() else {"total_tokens": 0, "per_node": {}}

        this_run = self.as_dict()
        for node, usage in this_run["per_node"].items():
            existing = merged["per_node"].setdefault(
                node, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
                existing[key] += usage[key]

        merged["total_tokens"] = sum(u["total_tokens"] for u in merged["per_node"].values())
        merged["budget"] = this_run["budget"]
        prompt = sum(u["prompt_tokens"] for u in merged["per_node"].values())
        completion = sum(u["completion_tokens"] for u in merged["per_node"].values())
        merged["estimated_cost_usd"] = round(
            prompt / 1_000_000 * _COST_PER_MILLION["default_prompt"]
            + completion / 1_000_000 * _COST_PER_MILLION["default_completion"],
            4,
        )
        path.write_text(json.dumps(merged, indent=2))


def _extract_usage(response: Any) -> tuple[int, int]:
    """Best-effort extraction across Anthropic/OpenAI-shaped llm_output payloads."""
    llm_output = getattr(response, "llm_output", None) or {}

    usage = llm_output.get("usage")
    if usage:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    token_usage = llm_output.get("token_usage")
    if token_usage:
        return int(token_usage.get("prompt_tokens", 0)), int(token_usage.get("completion_tokens", 0))

    # Fall back to per-generation usage_metadata (langchain-core >=0.2 message attr).
    prompt = completion = 0
    for gen_list in getattr(response, "generations", []) or []:
        for gen in gen_list:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if meta:
                prompt += int(meta.get("input_tokens", 0))
                completion += int(meta.get("output_tokens", 0))
    return prompt, completion
