"""GUI chat mode: watch the agent's investigation live and approve/reject/
send-back-for-more-evidence at the checkpoint from a browser instead of the
CLI. Same LangGraph state machine as `agent.cli` — this is just a different
front end driving the same `interrupt()`/`Command(resume=...)` contract.

Run with:
    streamlit run chat_ui/app.py
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from agent.graph import build_state_graph, open_checkpointer
from agent.memory import ProceduralMemory, open_memory_store
from agent.token_tracker import TokenTracker
from agent.trail import Trail
from config.schema import ProjectConfig

load_dotenv()

RUNS_ROOT = _PROJECT_ROOT / "data" / "runs"
MEMORY_STORE_PATH = _PROJECT_ROOT / "data" / "memory" / "store.sqlite"

st.set_page_config(page_title="RTL-Lens", page_icon="🔍", layout="wide")


def _init_state() -> None:
    defaults = {
        "run_id": None,
        "config_path": "config/tpe.yaml",
        "failure_dir": "",
        "pending_interrupt": None,
        "finished": None,
        "token_usage": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _render_message(msg) -> None:
    if isinstance(msg, SystemMessage):
        return
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            if msg.content:
                st.write(msg.content)
            for tc in getattr(msg, "tool_calls", None) or []:
                st.code(f"{tc['name']}({json.dumps(tc['args'])})", language="text")
    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant", avatar="🔧"):
            content = str(msg.content)
            st.text(content[:1200] + ("..." if len(content) > 1200 else ""))


def _run_graph(resume_value: dict | None = None) -> None:
    run_id = st.session_state.run_id
    run_dir = RUNS_ROOT / run_id
    config = ProjectConfig.load(st.session_state.config_path)
    tracker = TokenTracker(budget=config.budget)
    trail = Trail(run_dir)
    trail.append("resumed" if resume_value else "run_started", decision=resume_value)

    with open_memory_store(MEMORY_STORE_PATH) as store, open_checkpointer(run_dir) as checkpointer:
        memory = ProceduralMemory(store, config.project)
        graph = build_state_graph(config, run_dir, tracker, procedural_memory=memory).compile(
            checkpointer=checkpointer
        )
        thread_cfg = {"configurable": {"thread_id": run_id}}

        if resume_value is None:
            input_ = {
                "run_id": run_id,
                "config_path": st.session_state.config_path,
                "failure_dir": st.session_state.failure_dir,
                "run_dir": str(run_dir),
            }
        else:
            input_ = Command(resume=resume_value)

        transcript = st.container()
        rendered = 0
        result: dict = {}
        for event in graph.stream(input_, config=thread_cfg, stream_mode="values"):
            result = event
            msgs = event.get("messages", [])
            with transcript:
                for m in msgs[rendered:]:
                    _render_message(m)
            rendered = len(msgs)

        tracker.write_report(run_dir)
        st.session_state.token_usage = tracker.as_dict()
        status = trail.record(result)

        if status == "awaiting_approval":
            st.session_state.pending_interrupt = result["__interrupt__"][0].value
            st.session_state.finished = None
        else:
            st.session_state.pending_interrupt = None
            st.session_state.finished = result


_init_state()

st.title("🔍 RTL-Lens")
st.caption("Graph-grounded RTL debugging — watch the investigation, evidence, and fix as they happen.")

with st.sidebar:
    st.header("New diagnosis")
    st.session_state.config_path = st.text_input("Config", st.session_state.config_path)
    st.session_state.failure_dir = st.text_input(
        "Failure directory", st.session_state.failure_dir,
        placeholder="/path/to/SIM_LOGS/WORK/smoke/<block>.<test>/rtl_sim",
    )
    start = st.button("Start diagnosis", type="primary", disabled=not st.session_state.failure_dir)

    if st.session_state.token_usage:
        st.divider()
        st.subheader("Token usage")
        tu = st.session_state.token_usage
        st.metric("Total tokens", tu["total_tokens"])
        st.metric("Est. cost", f"${tu['estimated_cost_usd']:.4f}")
        for node, usage in tu["per_node"].items():
            st.caption(f"{node}: {usage['total_tokens']} tokens ({usage['calls']} calls)")

if start:
    st.session_state.run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_ROOT / st.session_state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": st.session_state.run_id,
                "config_path": st.session_state.config_path,
                "failure_dir": st.session_state.failure_dir,
            },
            indent=2,
        )
    )
    st.session_state.pending_interrupt = None
    st.session_state.finished = None
    with st.spinner("Investigating..."):
        _run_graph()

if st.session_state.run_id:
    st.caption(f"Run ID: `{st.session_state.run_id}`")

if st.session_state.pending_interrupt:
    payload = st.session_state.pending_interrupt
    st.subheader("🔎 Diagnosis ready — awaiting approval")

    label = payload["confidence_label"]
    color = {"high": "green", "medium": "orange", "low": "red"}.get(label, "gray")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Confidence:** :{color}[{payload['confidence_score']:.2f} ({label})]")
        st.caption(payload["confidence_rationale"])
        st.markdown(f"**Root cause:** `{payload['file']}:{payload.get('line')}`")
        st.write(payload["explanation"])
    with col2:
        st.code(payload["patch_diff"], language="diff")

    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Approve", type="primary", use_container_width=True):
        with st.spinner("Applying patch and rerunning test..."):
            _run_graph(resume_value={"decision": "approve"})
        st.rerun()
    if c2.button("❌ Reject", use_container_width=True):
        _run_graph(resume_value={"decision": "reject"})
        st.rerun()
    if c3.button("🔁 More evidence", use_container_width=True):
        with st.spinner("Gathering more evidence..."):
            _run_graph(resume_value={"decision": "more_evidence"})
        st.rerun()

if st.session_state.finished:
    result = st.session_state.finished
    phase = result.get("phase")
    if phase == "verified":
        st.success(f"✅ Fix verified. Branch: `{result.get('branch_name')}`")
        st.text(result.get("rerun_summary", ""))
    elif phase == "failed":
        st.error(f"❌ Rerun failed after applying the fix. Branch: `{result.get('branch_name')}`")
        st.text(result.get("rerun_summary", ""))
    elif result.get("approval_decision") == "reject":
        st.warning("Patch rejected by reviewer. Run ended.")
