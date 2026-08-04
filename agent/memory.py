"""Two token-conservation mechanisms built on LangMem:

1. Short-term: `SummarizationNode` compresses the tool-call history once it
   grows past a budget, so late-investigation LLM calls aren't re-paying for
   every early exploratory query verbatim. The full, uncompressed transcript
   still lives in `state["messages"]` for the trail/dashboard — only the
   *model's view* (`summarized_messages`) shrinks.
2. Long-term: a persistent `SqliteStore` (LangGraph's store abstraction, the
   backend LangMem's memory tools are built on) holds one procedural-memory
   entry per (project, error_class) — which tool sequence found the root
   cause last time. `ProceduralMemory.recall` is consulted before
   investigation starts and injected as a short hint, so a repeat failure
   category needs fewer exploratory tool calls the second time around.

No embedding model is configured on the store (semantic search would need a
Voyage/OpenAI-style embeddings API key this project doesn't otherwise need) —
lookups are exact-match on (project, error_class), which is all this needs.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from langmem.short_term import SummarizationNode

from agent.logging_config import get_logger, log_event
from agent.state import DebugState, EvidenceItem

logger = get_logger("memory")

_NAMESPACE_PREFIX = "procedural"


@contextmanager
def open_memory_store(path: Path) -> Iterator[BaseStore]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteStore.from_conn_string(str(path)) as store:
        yield store


def build_summarization_node(model: BaseChatModel, max_tokens: int = 6000, max_summary_tokens: int = 256):
    """Compresses `messages` -> `summarized_messages` once the running token
    count exceeds `max_tokens`. Cheap no-op below that threshold.

    Wrapped in a try/except: LangMem's running-summary bookkeeping (tracked in
    `state["context"]`) has edge cases across an interrupt/resume boundary —
    specifically, resuming after the checkpoint (as `more-evidence` does) has
    been observed to produce an empty message list on a later summarization
    pass, which the underlying Anthropic call then rejects outright. That
    failure must never take down the whole investigation: falling back to
    passing `messages` through unsummarized costs some tokens, not correctness.
    """
    node = SummarizationNode(
        model=model,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
        input_messages_key="messages",
        output_messages_key="summarized_messages",
    )

    def safe_compress(state: DebugState) -> dict:
        try:
            return node.invoke(state)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "summarization_failed_falling_back", error=str(exc), level=30)
            return {"summarized_messages": state["messages"]}

    return safe_compress


class ProceduralMemory:
    def __init__(self, store: BaseStore, project: str):
        self.store = store
        self.namespace = (_NAMESPACE_PREFIX, project)

    def recall(self, error_class: str) -> str | None:
        item = self.store.get(self.namespace, error_class)
        if item is None:
            return None
        log_event(logger, "memory_recall_hit", error_class=error_class)
        return item.value.get("strategy_hint")

    def remember(self, error_class: str, test: str, evidence: list[EvidenceItem], root_cause_file: str) -> None:
        tool_sequence = [e.tool for e in evidence]
        hint = (
            f"A previous '{error_class}' failure ({test}) was root-caused in {root_cause_file}. "
            f"The effective investigation order was: {' -> '.join(tool_sequence) or '(direct)'}."
        )
        self.store.put(
            self.namespace,
            error_class,
            {"strategy_hint": hint, "last_test": test, "root_cause_file": root_cause_file},
        )
        log_event(logger, "memory_written", error_class=error_class, root_cause_file=root_cause_file)


def apply_recall_hint(state: DebugState, hint: str | None) -> dict:
    """Injected into the ingest_failure prompt when a matching prior run exists.

    A HumanMessage, not a second SystemMessage: Anthropic's API rejects
    non-consecutive system messages, and this is appended after the
    conversation's one leading SystemMessage + the failure HumanMessage.
    """
    if not hint:
        return {}
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=f"Memory from a prior similar failure: {hint}")]}
