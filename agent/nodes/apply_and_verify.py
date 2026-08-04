"""Only reached after human approval. Creates a fix branch, applies the
staged patch, commits it, and reruns the exact failing test to verify —
this is the only node that touches the RTL repo's working tree or git state.
"""
from __future__ import annotations

from pathlib import Path

from agent.logging_config import get_logger, log_event
from agent.state import DebugState
from agent.tools import git_tools, sim_tools
from config.schema import ProjectConfig

logger = get_logger("nodes.apply_and_verify")


def build_apply_and_verify_node(config: ProjectConfig, run_dir: Path):
    def apply_and_verify(state: DebugState) -> dict:
        hyp = state["hypothesis"]
        failure = state["failure"]
        branch_name = f"agent-fix/{failure.test.replace('.', '-')}-{state['run_id'][:8]}"
        repo_path = config.rtl_repo.path

        try:
            git_tools.create_branch(repo_path, branch_name)
            git_tools.apply_patch(repo_path, (run_dir / "patch.diff").resolve())
            sha = git_tools.commit(
                repo_path,
                f"agent: fix {failure.test}\n\n{hyp.explanation}\n\n"
                f"Diagnosed via graph traversal + spec, confidence={state['confidence'].score:.2f}.",
            )
            log_event(logger, "fix_committed", branch=branch_name, sha=sha)

            result = sim_tools.run_test(config.sim, repo_path, failure.test, seed=failure.seed)
            phase = "verified" if result.passed else "failed"
            log_event(logger, "rerun_result", passed=result.passed, phase=phase)
            return {
                "branch_name": branch_name,
                "rerun_passed": result.passed,
                "rerun_summary": result.signature or result.stdout_tail,
                "phase": phase,
            }
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "apply_failed", error=str(exc), level=40)
            return {
                "branch_name": branch_name,
                "rerun_passed": False,
                "rerun_summary": f"apply/verify error: {exc}",
                "phase": "failed",
            }

    return apply_and_verify
