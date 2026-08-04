"""Parses a failing test's log directory into a structured `Failure`.

Point this at a `.../<block>.<test>/rtl_sim` directory (or the `<block>.<test>`
parent — both are accepted) and it will read `status.json` (preferred,
machine-generated) and `results.xml` (JUnit) for the authoritative signature,
falling back to grepping `run.log` directly if neither is present.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from agent.logging_config import get_logger, log_event
from agent.state import Failure
from config.schema import SimConfig

logger = get_logger("tools.log")

_STOPWORDS = {
    "the", "and", "for", "with", "want", "got", "should", "alone", "leave", "set",
    "clearing", "value", "values", "mismatch", "mismatches", "expected", "row",
    "test", "error", "fatal", "assertion", "failed", "status", "clear",
}
_IDENT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b")
_PATH_LIKE_RE = re.compile(r"[./\\]")


_SEED_SUFFIX_RE = re.compile(r"\.seed(\d+)$")


def _guess_block_test(failure_dir: Path) -> tuple[str, str, int | None]:
    tag_dir = failure_dir if not failure_dir.name == "rtl_sim" else failure_dir.parent
    tag = tag_dir.name  # "<block>.<test>" or "<block>.<test>.seed<N>" for a seeded random run

    seed: int | None = None
    seed_match = _SEED_SUFFIX_RE.search(tag)
    if seed_match:
        seed = int(seed_match.group(1))
        tag = tag[: seed_match.start()]

    if "." in tag:
        block, test = tag.split(".", 1)
    else:
        block, test = "unknown", tag
    return block, test, seed


def _extract_signals(text: str) -> list[str]:
    """Best-effort candidate signal/register names from a failure message —
    the investigation loop uses this as a starting point for graph_search, not
    as ground truth, so false positives here are cheap and false negatives are
    recovered by the LLM reading the raw message itself."""
    candidates = {m.group(0) for m in _IDENT_RE.finditer(text)}
    signals = sorted(
        (
            c for c in candidates
            if "_" in c
            and c.lower() not in _STOPWORDS
            and not _PATH_LIKE_RE.search(c)
        ),
        key=len,
        reverse=True,
    )
    return signals[:15]


def parse_failure(failure_dir: Path, sim_cfg: SimConfig | None = None) -> Failure:
    rtl_sim_dir = failure_dir if failure_dir.name == "rtl_sim" else failure_dir / "rtl_sim"
    if not rtl_sim_dir.is_dir():
        rtl_sim_dir = failure_dir  # caller pointed straight at the log dir

    block, test, seed = _guess_block_test(failure_dir)

    status = _read_json(rtl_sim_dir / "status.json")
    run_log = _read_text(rtl_sim_dir / "run.log")
    results_xml_summary = _summarize_results_xml(rtl_sim_dir / "results.xml")

    message = ""
    error_class = "Unknown"
    if status and status.get("signature"):
        message = status["signature"]
    elif results_xml_summary:
        message = results_xml_summary
    elif run_log:
        message = _tail_error_line(run_log)

    m = re.search(r"([A-Za-z_][A-Za-z0-9_.]*Error|UVM\w+)", message)
    if m:
        error_class = m.group(1)

    raw_excerpt = "\n".join(run_log.splitlines()[-60:]) if run_log else message

    failure = Failure(
        test=f"{block}.{test}",
        block=block,
        work_dir=str(rtl_sim_dir),
        error_class=error_class,
        message=message,
        signals=_extract_signals(message),
        raw_excerpt=raw_excerpt,
        seed=seed,
    )
    log_event(logger, "failure_parsed", test=failure.test, error_class=error_class, signals=failure.signals, seed=seed)
    return failure


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_text(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def _tail_error_line(run_log: str) -> str:
    lines = run_log.splitlines()
    for line in reversed(lines):
        if any(k in line for k in ("Error", "FATAL", "AssertionError", "MismatchError")):
            return line.strip()
    return lines[-1].strip() if lines else ""


def _summarize_results_xml(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        from junitparser import JUnitXml
    except ImportError:
        return ""
    try:
        xml = JUnitXml.fromfile(str(path))
    except Exception:  # noqa: BLE001
        return ""
    for suite in xml:
        for case in suite:
            if case.result:
                for r in case.result:
                    if getattr(r, "message", None):
                        return r.message
    return ""
