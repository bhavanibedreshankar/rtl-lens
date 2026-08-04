"""Wires the LangGraph state machine: ingest -> investigate loop -> synthesize
-> confidence gate (may bounce back to investigate) -> human checkpoint
(interrupt) -> apply & verify.

`build_state_graph` returns an uncompiled graph; `open_checkpointer` is a
context manager yielding a `SqliteSaver` backed by a file inside the run's own
directory, so a diagnose-phase CLI invocation can exit at the checkpoint and a
later, separate `apply` invocation can resume the exact same run from disk.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.memory import ProceduralMemory, apply_recall_hint, build_summarization_node
from agent.nodes.apply_and_verify import build_apply_and_verify_node
from agent.nodes.checkpoint import build_checkpoint_node, route_after_checkpoint
from agent.nodes.ingest_failure import ingest_failure
from agent.nodes.investigate import (
    build_investigate_node,
    build_tools_node,
    route_after_investigate,
)
from agent.nodes.score_confidence import (
    request_more_evidence,
    route_after_confidence,
    score_confidence,
)
from agent.nodes.synthesize_hypothesis import build_synthesize_node
from agent.state import DebugState
from agent.token_tracker import TokenTracker
from agent.tools.graph_tools import build_graph_tools
from agent.tools.rtl_tools import build_rtl_tools
from agent.tools.spec_tools import build_spec_tools
from config.schema import ProjectConfig


def build_state_graph(
    config: ProjectConfig,
    run_dir: Path,
    token_tracker: TokenTracker,
    procedural_memory: ProceduralMemory | None = None,
) -> StateGraph:
    tools = (
        build_graph_tools(config.graph_db)
        + build_spec_tools(config.rtl_repo)
        + build_rtl_tools(config.rtl_repo)
    )

    # temperature=0 keeps tool-call argument selection deterministic during
    # investigation, where precision (exact signal/file names) matters more than
    # variety. The synthesis model rejects an explicit temperature outright for
    # some model families, so it's left at its default.
    planning_model = ChatAnthropic(model=config.models.planning, temperature=0).with_config(
        {"callbacks": [token_tracker], "metadata": {"langgraph_node": "investigate"}}
    )
    synthesis_model = ChatAnthropic(model=config.models.synthesis).with_config(
        {"callbacks": [token_tracker], "metadata": {"langgraph_node": "synthesize"}}
    )

    def _ingest_with_memory(state: DebugState) -> dict:
        result = ingest_failure(state, config=config)
        if procedural_memory is not None:
            hint = procedural_memory.recall(result["failure"].error_class)
            extra = apply_recall_hint(state, hint)
            if extra:
                result["messages"] = result["messages"] + extra["messages"]
        return result

    sg = StateGraph(DebugState)
    sg.add_node("ingest_failure", _ingest_with_memory)
    sg.add_node("investigate", build_investigate_node(planning_model, tools, config.limits))
    sg.add_node("tools", build_tools_node(tools))
    sg.add_node("compress_evidence", build_summarization_node(planning_model))
    sg.add_node("synthesize", build_synthesize_node(synthesis_model, config.rtl_repo))
    sg.add_node("score_confidence", score_confidence)
    sg.add_node("need_more_evidence", request_more_evidence)
    sg.add_node("checkpoint", build_checkpoint_node(run_dir))
    sg.add_node("apply_and_verify", build_apply_and_verify_node(config, run_dir))

    sg.add_edge(START, "ingest_failure")
    sg.add_edge("ingest_failure", "investigate")
    sg.add_conditional_edges(
        "investigate",
        lambda s: route_after_investigate(s, config.limits),
        {"tools": "tools", "synthesize": "synthesize"},
    )
    sg.add_edge("tools", "compress_evidence")
    sg.add_edge("compress_evidence", "investigate")
    sg.add_edge("synthesize", "score_confidence")
    sg.add_conditional_edges(
        "score_confidence",
        lambda s: route_after_confidence(s, config.limits),
        {"need_more_evidence": "need_more_evidence", "checkpoint": "checkpoint"},
    )
    sg.add_edge("need_more_evidence", "investigate")
    sg.add_conditional_edges(
        "checkpoint",
        route_after_checkpoint,
        {"approve": "apply_and_verify", "reject": END, "more_evidence": "need_more_evidence"},
    )
    sg.add_edge("apply_and_verify", END)

    return sg


@contextmanager
def open_checkpointer(run_dir: Path) -> Iterator[SqliteSaver]:
    """Deliberately uses JsonPlusSerializer's default (permissive) allowlist.

    An earlier version passed an explicit `allowed_msgpack_modules` list here
    to silence a deprecation warning about our pydantic state models — but
    passing *any* explicit list switches the serializer from "allow
    everything" to "allow only what's listed," which silently dropped
    langchain_core's own message classes (AIMessage, HumanMessage, ...) on
    resume and broke `more-evidence`/`approve` with a cryptic "at least one
    message is required" error. The warning is cosmetic; a broken resume
    is not — leaving the default alone is the correct tradeoff."""
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(run_dir / "checkpoint.sqlite"), check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()
