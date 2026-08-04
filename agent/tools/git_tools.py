"""Git operations against the RTL repo, used only by `apply_and_verify` — i.e.
only after a human has approved the patch at the checkpoint. Every fix lands
on its own branch; main is never touched directly.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from agent.logging_config import get_logger, log_event

logger = get_logger("tools.git")


class GitError(RuntimeError):
    pass


def _run(repo_path: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo_path, capture_output=True, text=True)
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def ensure_clean_working_tree(repo_path: Path) -> None:
    status = _run(repo_path, "status", "--porcelain")
    if status:
        raise GitError(
            "RTL repo working tree is not clean — refusing to create a fix branch on top of "
            "unrelated uncommitted changes:\n" + status
        )


def current_branch(repo_path: Path) -> str:
    return _run(repo_path, "rev-parse", "--abbrev-ref", "HEAD")


def create_branch(repo_path: Path, branch_name: str, base: str = "main") -> str:
    ensure_clean_working_tree(repo_path)
    _run(repo_path, "fetch", "--quiet") if _has_remote(repo_path) else None
    _run(repo_path, "checkout", base)
    existing = _run(repo_path, "branch", "--list", branch_name)
    if existing:
        raise GitError(f"branch '{branch_name}' already exists — pick a different name or reuse it explicitly")
    _run(repo_path, "checkout", "-b", branch_name)
    log_event(logger, "branch_created", branch=branch_name, base=base)
    return branch_name


def _has_remote(repo_path: Path) -> bool:
    return bool(_run(repo_path, "remote"))


def apply_patch(repo_path: Path, patch_path: Path) -> None:
    _run(repo_path, "apply", "--check", str(patch_path))
    _run(repo_path, "apply", str(patch_path))
    log_event(logger, "patch_applied", patch=str(patch_path))


def commit(repo_path: Path, message: str, paths: list[str] | None = None) -> str:
    if paths:
        _run(repo_path, "add", *paths)
    else:
        _run(repo_path, "add", "-A")
    _run(repo_path, "commit", "-m", message)
    sha = _run(repo_path, "rev-parse", "--short", "HEAD")
    log_event(logger, "committed", sha=sha, commit_message=message)
    return sha


def diff_against(repo_path: Path, base: str = "main") -> str:
    return _run(repo_path, "diff", base)
