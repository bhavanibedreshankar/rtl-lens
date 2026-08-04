"""Append-only JSONL event log per run — the evidence the dashboard is built
from. Deliberately separate from `logging_config`'s log file: the trail is a
structured, dashboard-consumable record of *what the agent concluded*, not a
debug log of *how it got there* (that's `agent.log.jsonl`).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Trail:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "trail.jsonl"

    def append(self, event_type: str, **fields: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), "event": event_type, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def write_summary(self, summary: dict) -> None:
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    def record(self, result: dict) -> str:
        """Shared by every front end (CLI, Streamlit) so the dashboard sees the
        same event shape regardless of which one drove a given run. Returns a
        short status string the caller can use for its own presentation.

        `result` is a LangGraph state snapshot — either mid-run (containing
        `__interrupt__` at the checkpoint) or the final state after END.
        """
        failure = result.get("failure")
        evidence = result.get("evidence") or []
        evidence_dump = [{"tool": e.tool, "args": e.args, "summary": e.summary} for e in evidence]
        suite = suite_from_failure(failure)

        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            self.append(
                "awaiting_approval",
                **{k: v for k, v in payload.items() if k != "patch_diff"},
                suite=suite,
                fail_signature=failure.message if failure else None,
                error_class=failure.error_class if failure else None,
                evidence=evidence_dump,
            )
            return "awaiting_approval"

        phase = result.get("phase")
        if phase in ("verified", "failed"):
            self.append(
                "run_finished",
                phase=phase,
                rerun_passed=result.get("rerun_passed"),
                branch=result.get("branch_name"),
                summary=result.get("rerun_summary"),
            )
            return phase
        if phase == "rejected" or result.get("approval_decision") == "reject":
            self.append("run_finished", phase="rejected")
            return "rejected"

        self.append("run_finished", phase=phase or "unknown")
        return phase or "unknown"


def suite_from_failure(failure) -> str:
    """failure.work_dir is `.../<suite>/<block>.<test>/rtl_sim` for a suite-run
    ingestion (e.g. `smoke`), so the suite name is two levels up."""
    if failure is None:
        return "unknown"
    parts = Path(failure.work_dir).parts
    return parts[-3] if len(parts) >= 3 else "unknown"
