"""The human-in-the-loop gate. Runs after a hypothesis has cleared the
confidence bar. Uses LangGraph's dynamic `interrupt()` so the exact same
graph works when driven from the CLI (blocks on stdin) or from the Streamlit
chat UI (renders Approve/Reject/More-evidence buttons and resumes with
`Command(resume=...)`).
"""
from __future__ import annotations

from pathlib import Path

from langgraph.types import interrupt

from agent.logging_config import get_logger, log_event
from agent.state import DebugState
from agent.tools.rtl_tools import stage_patch

logger = get_logger("nodes.checkpoint")


def build_checkpoint_node(run_dir: Path):
    def checkpoint(state: DebugState) -> dict:
        hyp = state["hypothesis"]
        conf = state["confidence"]
        stage_patch(run_dir, hyp.patch_diff)

        payload = {
            "test": state["failure"].test,
            "file": hyp.file,
            "line": hyp.line,
            "explanation": hyp.explanation,
            "confidence_score": conf.score,
            "confidence_label": conf.label,
            "confidence_rationale": conf.rationale,
            "patch_diff": hyp.patch_diff,
            "patch_path": str(run_dir / "patch.diff"),
        }
        log_event(logger, "awaiting_approval", **{k: v for k, v in payload.items() if k != "patch_diff"})

        decision = interrupt(payload)  # {"decision": "approve" | "reject" | "more_evidence"}
        decision_value = decision.get("decision", "reject") if isinstance(decision, dict) else str(decision)
        log_event(logger, "approval_decision", decision=decision_value)
        return {"approval_decision": decision_value, "phase": "awaiting_approval"}

    return checkpoint


def route_after_checkpoint(state: DebugState) -> str:
    decision = state.get("approval_decision")
    if decision == "approve":
        return "approve"
    if decision == "more_evidence":
        return "more_evidence"
    return "reject"
