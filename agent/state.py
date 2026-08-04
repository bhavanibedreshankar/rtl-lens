"""LangGraph state schema for a single debug session (one failing test)."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Failure(BaseModel):
    """A structured failure extracted from a test's log directory."""

    test: str
    block: str
    work_dir: str
    error_class: str = "Unknown"
    message: str = ""
    signals: list[str] = Field(default_factory=list)
    raw_excerpt: str = ""
    seed: int | None = None


class EvidenceItem(BaseModel):
    """One distilled tool call result, kept small to conserve tokens."""

    tool: str
    args: dict = Field(default_factory=dict)
    summary: str


class HypothesisDraft(BaseModel):
    """What the LLM produces directly. Deliberately NOT a raw unified diff —
    `old_lines`/`new_lines` let the code construct a correctly-headered patch
    mechanically (see `agent.tools.rtl_tools.build_unified_diff`), instead of
    trusting model-authored hunk line-count math, which was a recurring
    source of `git apply`-rejected "corrupt patch" failures."""

    file: str
    start_line: int = Field(description="1-indexed line where old_lines begins in the file")
    old_lines: list[str] = Field(
        description="The exact current source lines being replaced, verbatim as read via rtl_read_file "
        "(no line-number prefixes, no added/removed whitespace)"
    )
    new_lines: list[str] = Field(description="The replacement lines")
    explanation: str
    self_reported_confidence: float = 0.5


class Hypothesis(BaseModel):
    """HypothesisDraft plus the mechanically-constructed patch_diff."""

    file: str
    line: int | None = None
    explanation: str
    patch_diff: str
    self_reported_confidence: float = 0.5


class ConfidenceResult(BaseModel):
    score: float
    rationale: str
    heuristics: dict = Field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.score >= 0.75:
            return "high"
        if self.score >= 0.45:
            return "medium"
        return "low"


Phase = Literal["diagnosing", "awaiting_approval", "applying", "verified", "rejected", "failed"]


class DebugState(TypedDict, total=False):
    run_id: str
    config_path: str
    failure_dir: str
    run_dir: str

    messages: Annotated[list[BaseMessage], add_messages]
    # LangMem SummarizationNode's working state: running summary + the
    # (possibly shorter) message list actually sent to the LLM. `messages`
    # itself is never truncated — it's the full audit trail for the dashboard.
    context: dict
    summarized_messages: list[BaseMessage]

    failure: Failure
    evidence: list[EvidenceItem]
    investigation_steps: int

    hypothesis: Hypothesis | None
    confidence: ConfidenceResult | None
    confidence_retries: int

    phase: Phase
    approval_decision: Literal["approve", "reject", "more_evidence"] | None

    branch_name: str | None
    rerun_passed: bool | None
    rerun_summary: str | None

    token_usage: dict
