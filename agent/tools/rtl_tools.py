"""Scoped, read-mostly access to RTL source for the diagnosis loop.

Two hard boundaries enforced here, not just documented:
1. Path scoping — only files under `rtl_repo.rtl_dir` are readable. Anything
   else (docs, testbenches, the bug_list.md answer key) is out of reach for
   this tool, independent of `excluded_paths`.
2. `excluded_paths` (the blind-debugging guarantee) is checked in addition,
   so it also covers files that happen to live under rtl_dir in a future
   project's layout.

Writing a patch to the actual working tree is *not* done here — that only
happens post-approval, in `git_tools.apply_patch`. This module's write-side
(`stage_patch`) only ever writes into the run's own `data/runs/<id>/` dir.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.logging_config import get_logger, log_event
from config.schema import RtlRepoConfig

logger = get_logger("tools.rtl")

_DEFAULT_WINDOW = 40  # lines of context shown around a requested line


class ScopeError(RuntimeError):
    pass


def _resolve_in_scope(cfg: RtlRepoConfig, relative_path: str) -> Path:
    candidate = (cfg.path / relative_path).resolve()
    repo_root = cfg.path.resolve()
    rtl_root = cfg.rtl_root.resolve()

    if not str(candidate).startswith(str(repo_root) + "/"):
        raise ScopeError(f"'{relative_path}' resolves outside the RTL repo")
    if not str(candidate).startswith(str(rtl_root) + "/"):
        raise ScopeError(f"'{relative_path}' is outside {cfg.rtl_dir}/ — only RTL source is readable")
    if cfg.is_excluded(relative_path):
        raise ScopeError(f"'{relative_path}' is excluded from agent access")
    if not candidate.is_file():
        raise ScopeError(f"'{relative_path}' does not exist")
    return candidate


class ReadFileArgs(BaseModel):
    path: str = Field(description="Path to an RTL file, relative to the repo root, e.g. 'rtl/command_processor/tpe_cmd_proc.sv'")
    start_line: int | None = Field(default=None, description="1-indexed line to center the view on")
    end_line: int | None = Field(default=None, description="If set with start_line, show exactly this range instead of a window")


def build_rtl_tools(cfg: RtlRepoConfig) -> list[StructuredTool]:
    def read_rtl_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read RTL source, scoped to the design's rtl/ directory. Give start_line for a
        windowed view around a line of interest (e.g. from a graph_trace_driver result's
        file:line), or start_line+end_line for an exact range."""
        try:
            full_path = _resolve_in_scope(cfg, path)
        except ScopeError as exc:
            log_event(logger, "rtl_read_denied", path=path, reason=str(exc), level=30)
            return f"(denied) {exc}"

        lines = full_path.read_text().splitlines()
        n = len(lines)
        if start_line is not None:
            if end_line is not None:
                lo, hi = max(1, start_line), min(n, end_line)
            else:
                lo, hi = max(1, start_line - _DEFAULT_WINDOW // 2), min(n, start_line + _DEFAULT_WINDOW // 2)
        else:
            lo, hi = 1, min(n, _DEFAULT_WINDOW)

        log_event(logger, "rtl_read", path=path, lo=lo, hi=hi)
        numbered = "\n".join(f"{i:>5}: {lines[i - 1]}" for i in range(lo, hi + 1))
        return f"--- {path} (lines {lo}-{hi} of {n}) ---\n{numbered}"

    return [StructuredTool.from_function(read_rtl_file, name="rtl_read_file", args_schema=ReadFileArgs)]


def stage_patch(run_dir: Path, patch_diff: str) -> Path:
    """Write a proposed patch into the run's own directory only — never the RTL repo.

    Models reliably omit the final newline on generated diff text; `git apply`
    treats a unified diff whose last hunk line isn't newline-terminated as
    corrupt (not just "no trailing newline in the source"), so it's restored here.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    patch_path = run_dir / "patch.diff"
    content = patch_diff if patch_diff.endswith("\n") else patch_diff + "\n"
    patch_path.write_text(content)
    return patch_path
