"""Combines the model's self-reported confidence with cheap, deterministic
heuristic signals so confidence isn't just "the LLM's opinion of itself."
Low-confidence hypotheses are bounced back into investigation before ever
reaching the human checkpoint.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.logging_config import get_logger, log_event
from agent.state import ConfidenceResult, DebugState
from config.schema import LimitsConfig

logger = get_logger("nodes.score_confidence")

_MORE_EVIDENCE_PROMPT = (
    "Your confidence in that hypothesis was too low to proceed. Gather more evidence — "
    "trace the driver further back, check fanin, or re-read the spec — before restating "
    "your root cause."
)


def score_confidence(state: DebugState) -> dict:
    hyp = state["hypothesis"]
    evidence = state.get("evidence", [])

    file_mentioned = any(hyp.file in (e.summary + str(e.args)) for e in evidence)
    used_graph = any(e.tool.startswith("graph_") for e in evidence)
    used_spec = any(e.tool == "spec_search" for e in evidence)
    read_the_file = any(
        e.tool == "rtl_read_file" and e.args.get("path") == hyp.file for e in evidence
    )

    heuristics = {
        "file_appeared_in_evidence": file_mentioned,
        "used_graph_traversal": used_graph,
        "consulted_spec": used_spec,
        "read_patched_file_directly": read_the_file,
    }
    grounded_bonus = (
        0.4 * file_mentioned + 0.3 * used_graph + 0.15 * used_spec + 0.15 * read_the_file
    )
    score = max(0.0, min(1.0, 0.6 * hyp.self_reported_confidence + 0.4 * grounded_bonus))

    rationale_bits = [f"model self-reported {hyp.self_reported_confidence:.2f}"]
    rationale_bits += [k.replace("_", " ") for k, v in heuristics.items() if v]
    rationale = "; ".join(rationale_bits)

    result = ConfidenceResult(score=score, rationale=rationale, heuristics=heuristics)
    log_event(logger, "confidence_scored", score=score, label=result.label)
    return {"confidence": result}


def route_after_confidence(state: DebugState, limits: LimitsConfig) -> str:
    confidence: ConfidenceResult = state["confidence"]
    retries = state.get("confidence_retries", 0)
    if confidence.score < limits.min_confidence_to_checkpoint and retries < limits.max_confidence_retries:
        return "need_more_evidence"
    return "checkpoint"


def request_more_evidence(state: DebugState) -> dict:
    return {
        "messages": [HumanMessage(content=_MORE_EVIDENCE_PROMPT)],
        "confidence_retries": state.get("confidence_retries", 0) + 1,
    }
