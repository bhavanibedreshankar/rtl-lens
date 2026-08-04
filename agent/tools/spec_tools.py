"""Read-only access to the design spec / register map — the "what's correct"
side of diagnosis, complementing the graph's "what's connected."

Scoped strictly to `rtl_repo.spec_paths` from config; nothing else under the
RTL repo's docs/ tree is reachable through this tool, which is what keeps
`docs/verification/bug_list.md` out of reach even though it lives in the same
directory tree as legitimate spec docs.
"""
from __future__ import annotations

import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.logging_config import get_logger, log_event
from config.schema import RtlRepoConfig

logger = get_logger("tools.spec")

_MAX_SECTIONS = 5
_MAX_SECTION_CHARS = 1200

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class Section(BaseModel):
    file: str
    heading: str
    body: str


def _split_markdown(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("(whole file)", text)]
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[start:end].strip()))
    return sections


def _split_yaml_top_level(text: str) -> list[tuple[str, str]]:
    """Cheap top-level-key chunking for register-map-style YAML, no schema assumed."""
    sections: list[tuple[str, str]] = []
    current_key, buf = None, []
    for line in text.splitlines():
        if line and not line.startswith((" ", "\t", "#")) and line.rstrip().endswith(":"):
            if current_key is not None:
                sections.append((current_key, "\n".join(buf)))
            current_key = line.rstrip().rstrip(":")
            buf = []
        else:
            buf.append(line)
    if current_key is not None:
        sections.append((current_key, "\n".join(buf)))
    return sections or [("(whole file)", text)]


def _load_sections(cfg: RtlRepoConfig) -> list[Section]:
    sections: list[Section] = []
    for rel in cfg.spec_paths:
        full = cfg.resolve(rel)
        if not full.is_file():
            log_event(logger, "spec_file_missing", path=rel, level=30)
            continue
        text = full.read_text()
        splitter = _split_yaml_top_level if rel.endswith((".yaml", ".yml")) else _split_markdown
        for heading, body in splitter(text):
            if body.strip():
                sections.append(Section(file=rel, heading=heading, body=body))
    return sections


class SpecSearchArgs(BaseModel):
    query: str = Field(description="Keyword, module name, register name, or behavior to look up in the spec/register map")


def build_spec_tools(cfg: RtlRepoConfig) -> list[StructuredTool]:
    # Loaded once per process; spec docs are static within a debug session.
    sections = _load_sections(cfg)

    def spec_search(query: str) -> str:
        """Search the design spec and register map for sections relevant to a keyword,
        module, register, or expected behavior. Use this to check what the RTL SHOULD do."""
        needle = query.lower()
        scored = []
        for s in sections:
            score = 0
            if needle in s.heading.lower():
                score += 2
            if needle in s.body.lower():
                score += 1
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        top = scored[:_MAX_SECTIONS]
        log_event(logger, "spec_search", query=query, hits=len(scored))
        if not top:
            return f"(no spec sections matched '{query}')"
        chunks = []
        for _, s in top:
            body = s.body[:_MAX_SECTION_CHARS]
            if len(s.body) > _MAX_SECTION_CHARS:
                body += " ...(truncated)"
            chunks.append(f"### [{s.file}] {s.heading}\n{body}")
        return "\n\n".join(chunks)

    return [StructuredTool.from_function(spec_search, name="spec_search", args_schema=SpecSearchArgs)]
