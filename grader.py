"""Standalone, offline scorer — deliberately never imported by anything under
`agent/`. Run this *after* a diagnosis run has reached its checkpoint, to
compare the agent's independently-reached root cause against
`docs/verification/bug_list.md`'s answer key. This is what lets the dashboard
claim "the agent found it independently" rather than "the agent read the
answer key" — the diagnosis loop's tool scoping (see
`agent/tools/rtl_tools.py::_resolve_in_scope` and
`agent/tools/spec_tools.py`) makes bug_list.md unreachable during the run;
this script is the only thing in the repo allowed to read it.

Usage:
    python grader.py --run-id <id> --bug-list /path/to/bug_list.md
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BugEntry:
    number: int
    block: str
    file: str
    lines: list[int]
    caught_by: str
    category: str


@dataclass
class GradeResult:
    run_id: str
    test: str
    agent_file: str
    agent_line: int | None
    expected_bug_number: int | None
    expected_file: str | None
    expected_lines: list[int] = field(default_factory=list)
    file_match: bool = False
    line_match: bool = False
    verdict: str = "no_matching_bug_entry"  # correct | file_only | incorrect | no_matching_bug_entry


def parse_bug_list(path: Path) -> list[BugEntry]:
    entries: list[BugEntry] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= {"|", "-", " ", ":"}:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != 7 or not cells[0].isdigit():
            continue
        num, block, file_line, _symptom, _root_cause, caught_by, category = cells
        file_line = file_line.strip("`")
        if ":" in file_line:
            file_part, lines_part = file_line.split(":", 1)
            lines = [int(x) for x in re.findall(r"\d+", lines_part)]
        else:
            file_part, lines = file_line, []
        entries.append(
            BugEntry(
                number=int(num), block=block, file=file_part.strip(),
                lines=lines, caught_by=caught_by, category=category,
            )
        )
    return entries


def _norm(path: str) -> str:
    return path.strip().lstrip("./")


def _last_diagnosis_event(run_dir: Path) -> dict:
    trail_path = run_dir / "trail.jsonl"
    if not trail_path.is_file():
        raise SystemExit(f"no trail.jsonl found in {run_dir}")
    events = [json.loads(l) for l in trail_path.read_text().splitlines() if l.strip()]
    approvals = [e for e in events if e.get("event") == "awaiting_approval"]
    if not approvals:
        raise SystemExit(f"run {run_dir.name} never reached a diagnosis — nothing to grade")
    return approvals[-1]


def grade(run_dir: Path, bug_list_path: Path) -> GradeResult:
    diagnosis = _last_diagnosis_event(run_dir)
    test = diagnosis["test"]
    agent_file = diagnosis["file"]
    agent_line = diagnosis.get("line")

    bugs = parse_bug_list(bug_list_path)
    test_name = test.split(".", 1)[-1]
    matches = [b for b in bugs if test_name in b.caught_by]

    if not matches:
        return GradeResult(
            run_id=run_dir.name, test=test, agent_file=agent_file, agent_line=agent_line,
            expected_bug_number=None, expected_file=None,
        )

    bug = matches[0]
    file_match = _norm(agent_file) == _norm(bug.file)
    if bug.lines and agent_line:
        line_match = any(abs(agent_line - l) <= 3 for l in bug.lines)
    else:
        line_match = file_match

    if file_match and line_match:
        verdict = "correct"
    elif file_match:
        verdict = "file_only"
    else:
        verdict = "incorrect"

    return GradeResult(
        run_id=run_dir.name, test=test, agent_file=agent_file, agent_line=agent_line,
        expected_bug_number=bug.number, expected_file=bug.file, expected_lines=bug.lines,
        file_match=file_match, line_match=line_match, verdict=verdict,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bug-list", required=True, help="Path to docs/verification/bug_list.md")
    parser.add_argument("--runs-root", default="data/runs")
    args = parser.parse_args()

    run_dir = Path(args.runs_root) / args.run_id
    result = grade(run_dir, Path(args.bug_list))

    (run_dir / "grade.json").write_text(json.dumps(asdict(result), indent=2))

    print(f"Run:      {result.run_id}")
    print(f"Test:     {result.test}")
    print(f"Agent:    {result.agent_file}:{result.agent_line}")
    if result.expected_bug_number:
        print(f"Expected: bug #{result.expected_bug_number} — {result.expected_file}:{result.expected_lines}")
    else:
        print("Expected: (no bug_list.md entry catches this test)")
    print(f"Verdict:  {result.verdict.upper()}")


if __name__ == "__main__":
    main()
