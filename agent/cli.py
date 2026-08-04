"""CLI entry point.

    python -m agent.cli debug --failure-dir <path-to-a-test's-rtl_sim-dir> --config config/tpe.yaml
    python -m agent.cli approve --run-id <id>
    python -m agent.cli reject --run-id <id>
    python -m agent.cli more-evidence --run-id <id>

`debug` runs the graph up to the human-approval checkpoint and stops —
LangGraph's `interrupt()` genuinely pauses execution there, persisted to a
per-run SQLite checkpoint, so `approve`/`reject`/`more-evidence` can resume it
from a completely separate process invocation (this is what "autonomous with
checkpoints" means in practice: unattended up to the gate, then resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from agent.graph import build_state_graph, open_checkpointer
from agent.logging_config import bind_run, get_logger, setup_logging
from agent.memory import ProceduralMemory, open_memory_store
from agent.token_tracker import TokenTracker
from agent.trail import Trail, suite_from_failure
from config.schema import ProjectConfig

logger = get_logger("cli")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Absolute paths: apply_and_verify's git/sim subprocesses run with cwd set to the
# *target RTL repo*, not this repo, so a relative run_dir would resolve wrong there.
RUNS_ROOT = _PROJECT_ROOT / "data" / "runs"
MEMORY_STORE_PATH = _PROJECT_ROOT / "data" / "memory" / "store.sqlite"


def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _load_meta(run_id: str) -> dict:
    return json.loads((_run_dir(run_id) / "meta.json").read_text())


def _thread_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _report(result: dict, trail: Trail, tracker: TokenTracker, run_id: str) -> None:
    status = trail.record(result)
    failure = result.get("failure")
    evidence = result.get("evidence") or []

    if status == "awaiting_approval":
        interrupts = result["__interrupt__"]
        payload = interrupts[0].value
        print("\n=== Diagnosis ready — awaiting approval ===")
        print(f"Run ID:      {run_id}")
        print(f"Suite:       {suite_from_failure(failure)}")
        print(f"Test:        {payload['test']}")
        if failure:
            print(f"Fail signature: {failure.message}")
        print(f"Root cause:  {payload['file']}:{payload.get('line')}")
        print(f"Explanation: {payload['explanation']}")
        print(f"Confidence:  {payload['confidence_score']:.2f} ({payload['confidence_label']}) — {payload['confidence_rationale']}")
        print(f"Investigation steps: {len(evidence)}")
        print(f"Patch:       {payload['patch_path']}")
        print(f"Tokens used: {tracker.total_tokens} (~${tracker.estimated_cost_usd():.4f})")
        print("\nReview the patch, then run one of:")
        print(f"  python -m agent.cli approve       --run-id {run_id}")
        print(f"  python -m agent.cli reject        --run-id {run_id}")
        print(f"  python -m agent.cli more-evidence --run-id {run_id}")
        return

    if status in ("verified", "failed"):
        print(f"\n=== Run finished: {status} ===")
        print(f"Branch:  {result.get('branch_name')}")
        print(f"Rerun passed: {result.get('rerun_passed')}")
        print(f"Summary: {result.get('rerun_summary')}")
        print(f"Tokens used: {tracker.total_tokens} (~${tracker.estimated_cost_usd():.4f})")
    elif status == "rejected":
        print("\nPatch rejected by reviewer. Run ended.")
    else:
        print(f"\nRun ended in state: phase={status}")


def cmd_debug(args: argparse.Namespace) -> None:
    load_dotenv()
    run_id = args.run_id or uuid.uuid4().hex[:12]
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir, level=os.environ.get("LOG_LEVEL", "INFO"))
    bind_run(run_id)

    config = ProjectConfig.load(args.config)
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": run_id, "config_path": args.config, "failure_dir": args.failure_dir}, indent=2)
    )

    trail = Trail(run_dir)
    trail.append("run_started", failure_dir=args.failure_dir, config=args.config)

    tracker = TokenTracker(budget=config.budget)

    with open_memory_store(MEMORY_STORE_PATH) as store, open_checkpointer(run_dir) as checkpointer:
        memory = ProceduralMemory(store, config.project)
        graph = build_state_graph(config, run_dir, tracker, procedural_memory=memory).compile(
            checkpointer=checkpointer
        )
        initial_state = {
            "run_id": run_id,
            "config_path": args.config,
            "failure_dir": args.failure_dir,
            "run_dir": str(run_dir),
        }
        result = graph.invoke(initial_state, config=_thread_config(run_id))
        tracker.write_report(run_dir)
        _report(result, trail, tracker, run_id)


def _resume(run_id: str, decision: str, config_override: str | None) -> None:
    load_dotenv()
    run_dir = _run_dir(run_id)
    meta = _load_meta(run_id)
    config = ProjectConfig.load(config_override or meta["config_path"])

    setup_logging(run_dir, level=os.environ.get("LOG_LEVEL", "INFO"))
    bind_run(run_id)

    tracker = TokenTracker(budget=config.budget)
    trail = Trail(run_dir)
    trail.append("resumed", decision=decision)

    with open_memory_store(MEMORY_STORE_PATH) as store, open_checkpointer(run_dir) as checkpointer:
        memory = ProceduralMemory(store, config.project)
        graph = build_state_graph(config, run_dir, tracker, procedural_memory=memory).compile(
            checkpointer=checkpointer
        )
        result = graph.invoke(Command(resume={"decision": decision}), config=_thread_config(run_id))
        tracker.write_report(run_dir)

        if decision == "approve" and result.get("phase") == "verified":
            hyp, failure = result.get("hypothesis"), result.get("failure")
            if hyp and failure:
                memory.remember(failure.error_class, failure.test, result.get("evidence", []), hyp.file)

        _report(result, trail, tracker, run_id)


def cmd_approve(args: argparse.Namespace) -> None:
    _resume(args.run_id, "approve", args.config)


def cmd_reject(args: argparse.Namespace) -> None:
    _resume(args.run_id, "reject", args.config)


def cmd_more_evidence(args: argparse.Namespace) -> None:
    _resume(args.run_id, "more_evidence", args.config)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m agent.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_debug = sub.add_parser("debug", help="Diagnose a failing test from its log directory")
    p_debug.add_argument("--failure-dir", required=True, help="Path to the test's rtl_sim (or parent) log directory")
    p_debug.add_argument("--config", required=True, help="Path to a project config YAML, e.g. config/tpe.yaml")
    p_debug.add_argument("--run-id", default=None, help="Reuse a specific run id (default: random)")
    p_debug.set_defaults(func=cmd_debug)

    for name, fn, help_text in (
        ("approve", cmd_approve, "Approve the staged patch, apply it, and rerun the test"),
        ("reject", cmd_reject, "Reject the staged patch and end the run"),
        ("more-evidence", cmd_more_evidence, "Send the agent back to gather more evidence"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-id", required=True)
        p.add_argument("--config", default=None, help="Override the config used at diagnose time")
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
