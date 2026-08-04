"""Structured logging for the debug agent.

Every run gets its own JSON-lines log file (alongside the trail written by
`agent.trail`) plus a human-readable console stream. Node modules should call
`get_logger(__name__)` rather than `logging.getLogger` directly so log records
carry a consistent `run_id` field once `bind_run` has been called.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

_run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)

_ROOT_LOGGER_NAME = "rtl_debug_agent"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": _run_id_var.get(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = _run_id_var.get()
        prefix = f"[{rid[:8]}] " if rid else ""
        base = f"{prefix}{record.levelname:<7} {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def bind_run(run_id: str) -> None:
    """Attach a run_id to every log record emitted in this context (thread/async task)."""
    _run_id_var.set(run_id)


def setup_logging(run_dir: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure the root agent logger. Safe to call multiple times (idempotent)."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ConsoleFormatter())
    logger.addHandler(console)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "agent.log.jsonl")
        file_handler.setFormatter(_JsonFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(_ROOT_LOGGER_NAME).getChild(name)


def log_event(logger: logging.Logger, message: str, level: int = logging.INFO, **fields) -> None:
    """Log with structured extra fields, surfaced in the JSON file handler."""
    logger.log(level, message, extra={"extra_fields": fields})
