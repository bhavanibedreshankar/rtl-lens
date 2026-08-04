"""Simulation invocation and result parsing.

Deliberately *not* exposed as an LLM-callable tool — rerunning a test is
expensive (tens of seconds of Verilator/cocotb) and only ever makes sense
once per verification attempt, driven by `agent.nodes.apply_and_verify`
after the human approval checkpoint, never speculatively by the model
during investigation.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel

from agent.logging_config import get_logger, log_event
from config.schema import SimConfig

logger = get_logger("tools.sim")

# run_sim's own layout (docs/flows/run_sim_flow.md): a single `-test` invocation
# (no `-suite`) always lands at $WORK_DIR/<work-dir-name>/<tag>/rtl_sim/, using
# "WORK" as the default work-dir-name — distinct from wherever a *suite* run
# (e.g. the original `-suite smoke` that produced the failure being diagnosed)
# put its results. Verification reruns always go through the `-test` path, so
# this fixed segment is where they'll be, regardless of `cfg.work_dir`'s own
# structure.
_DEFAULT_WORK_DIR_NAME = "WORK"


class SimResult(BaseModel):
    test: str
    passed: bool
    signature: str = ""
    wall_s: float | None = None
    returncode: int
    stdout_tail: str = ""


def run_test(cfg: SimConfig, repo_path: Path, test_tag: str, seed: int | None = None, timeout_s: int = 900) -> SimResult:
    cmd = cfg.run_cmd_template.format(test=test_tag)
    if seed is not None:
        cmd += f" -seed {seed}"
    # Point WORK_DIR at the same root the original failure was ingested from, so
    # this rerun's status.json lands somewhere we know to look (see module docstring).
    env = {**os.environ, "WORK_DIR": str(cfg.work_dir.parent.parent)}
    log_event(logger, "sim_run_start", cmd=cmd, cwd=str(repo_path), work_dir=env["WORK_DIR"])
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    status = _read_status(cfg, test_tag, seed)
    passed = status.get("state") == "PASS" if status else proc.returncode == 0
    result = SimResult(
        test=test_tag,
        passed=passed,
        signature=status.get("signature", "") if status else "",
        wall_s=status.get("duration_s") if status else None,
        returncode=proc.returncode,
        stdout_tail="\n".join((proc.stdout or "").splitlines()[-40:]),
    )
    log_event(logger, "sim_run_end", passed=passed, returncode=proc.returncode, status_found=status is not None)
    return result


def _read_status(cfg: SimConfig, test_tag: str, seed: int | None = None) -> dict | None:
    # A seeded rerun's output directory carries a `.seed<N>` suffix (run_sim's own
    # convention), distinct from the unseeded tag used to invoke `-test`.
    scope = f"{test_tag}.seed{seed}" if seed is not None else test_tag
    status_path = cfg.work_dir.parent.parent / _DEFAULT_WORK_DIR_NAME / scope / cfg.status_file
    if not status_path.is_file():
        return None
    try:
        return json.loads(status_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
