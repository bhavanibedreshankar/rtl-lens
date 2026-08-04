"""Pydantic schema for project adapter configs (config/*.yaml).

A project config is the *only* thing that needs to change to point this agent at a
different RTL design and graph database — no code in `agent/` should hardcode a path,
URL, or command. See `config/tpe.yaml` for a filled-in example.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class RtlRepoConfig(BaseModel):
    path: Path
    rtl_dir: str = "rtl"
    spec_paths: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)

    @property
    def rtl_root(self) -> Path:
        return self.path / self.rtl_dir

    def resolve(self, relative: str) -> Path:
        return self.path / relative

    def is_excluded(self, relative: str) -> bool:
        rel = str(Path(relative))
        return any(rel == str(Path(p)) for p in self.excluded_paths)


class GraphDbConfig(BaseModel):
    api_url: str
    local_fallback_cmd: str | None = None
    local_fallback_cwd: Path | None = None
    local_fallback_url: str | None = None
    request_timeout_s: float = 15.0


class SimConfig(BaseModel):
    run_cmd_template: str  # e.g. "./run_sim -test {test}"
    work_dir: Path
    status_file: str = "rtl_sim/status.json"
    run_log_file: str = "rtl_sim/run.log"
    results_xml_file: str = "rtl_sim/results.xml"


class ModelsConfig(BaseModel):
    planning: str = "claude-haiku-4-5-20251001"
    synthesis: str = "claude-sonnet-5"


class BudgetConfig(BaseModel):
    max_tokens_per_run: int = 150_000
    warn_at_pct: int = 80

    @field_validator("warn_at_pct")
    @classmethod
    def _pct_range(cls, v: int) -> int:
        if not (0 < v <= 100):
            raise ValueError("warn_at_pct must be in (0, 100]")
        return v


class LimitsConfig(BaseModel):
    max_investigation_steps: int = 6
    max_confidence_retries: int = 2
    min_confidence_to_checkpoint: float = 0.35


class ProjectConfig(BaseModel):
    project: str
    rtl_repo: RtlRepoConfig
    graph_db: GraphDbConfig
    sim: SimConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
