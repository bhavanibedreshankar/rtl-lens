"""LangChain tools over the RTL_Graph REST API.

Any RTL graph database wired in via config just needs to expose this same
contract (GET /search, /driver, /receivers, /fanin, /fanout, /path,
/signal/clock-domain, /signal/reset-tree, /module/hierarchy) — see
`RTL_Graph/src/api/main.py` for the reference implementation this was built
against, in particular the exact param names and response shapes (`fanin` ->
`{"fanin": [{"node": {...}, "distance": int}]}`, `driver` -> `{"drivers": [...]}`
etc.) that this module depends on.

Two token-conservation measures live here rather than in the LLM prompt:
1. An in-process cache keyed on (endpoint, params) so a repeated query within
   one investigation costs zero extra tokens and zero extra latency.
2. Result distillation — BFS-shaped payloads (fanin/fanout) are large; only a
   bounded, ranked summary is handed to the model, not the raw JSON.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.logging_config import get_logger, log_event
from config.schema import GraphDbConfig

logger = get_logger("tools.graph")

_MAX_NODES_IN_SUMMARY = 12
_MAX_JSON_CHARS = 3000


class GraphQueryError(RuntimeError):
    pass


class GraphDBClient:
    """Thin, cached HTTP client with local-fallback for the RTL_Graph API."""

    def __init__(self, cfg: GraphDbConfig):
        self.cfg = cfg
        self._base_url = cfg.api_url
        self._cache: dict[str, Any] = {}
        self._checked_health = False

    def _resolve_base_url(self) -> str:
        if self._checked_health:
            return self._base_url
        self._checked_health = True
        try:
            httpx.get(f"{self._base_url}/health", timeout=5.0).raise_for_status()
        except Exception as exc:
            if self.cfg.local_fallback_url:
                log_event(
                    logger,
                    "Primary graph DB unreachable, falling back to local instance",
                    level=30,
                    primary=self._base_url,
                    error=str(exc),
                )
                self._base_url = self.cfg.local_fallback_url
            else:
                raise
        return self._base_url

    def get(self, path: str, params: dict | None = None) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_key = f"{path}?{json.dumps(params, sort_keys=True)}"
        if cache_key in self._cache:
            log_event(logger, "cache_hit", endpoint=path, params=params)
            return self._cache[cache_key]

        base = self._resolve_base_url()
        try:
            resp = httpx.get(f"{base}{path}", params=params, timeout=self.cfg.request_timeout_s)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:  # noqa: BLE001
                detail = str(exc)
            data = {"not_found": True, "detail": detail}
            self._cache[cache_key] = data
            return data
        except httpx.HTTPError as exc:
            raise GraphQueryError(f"{path} failed: {exc}") from exc
        data = resp.json()
        self._cache[cache_key] = data
        log_event(logger, "graph_query", endpoint=path, params=params, cached=False)
        return data


def _short(node: dict) -> str:
    return f"{node.get('node_type', '?')}:{node.get('name', node.get('id', '?'))}"


def _distill_search(data: dict) -> str:
    if data.get("not_found"):
        return "(no matches)"
    results = data.get("results", [])
    if not results:
        return "(no matches)"
    shown = results[: _MAX_NODES_IN_SUMMARY]
    lines = [f"- {_short(n)} (id={n.get('id')}, module={n.get('module', n.get('attrs', {}).get('module'))})" for n in shown]
    text = "\n".join(lines)
    if len(results) > _MAX_NODES_IN_SUMMARY:
        text += f"\n... and {len(results) - _MAX_NODES_IN_SUMMARY} more (truncated, narrow your query)"
    return text


def _distill_fanin_fanout(data: dict, key: str) -> str:
    if data.get("not_found"):
        return f"(signal not found: {data.get('detail')})"
    items = data.get(key, [])
    if not items:
        return "(empty — no transitive dependencies found within max_depth)"
    items = sorted(items, key=lambda x: x.get("distance", 999))
    shown = items[:_MAX_NODES_IN_SUMMARY]
    lines = [f"- {_short(it['node'])} (distance={it.get('distance')})" for it in shown]
    text = "\n".join(lines)
    if len(items) > _MAX_NODES_IN_SUMMARY:
        text += f"\n... and {len(items) - _MAX_NODES_IN_SUMMARY} more (truncated, reduce max_depth)"
    return text


def _distill_driver(data: dict) -> str:
    if data.get("not_found"):
        return f"(signal not found: {data.get('detail')})"
    drivers = data.get("drivers", [])
    if not drivers:
        return "(no explicit driver found — may be a primary input/port)"
    lines = []
    for d in drivers:
        driver = d.get("driver", {})
        always = d.get("always_block")
        loc = (always or driver).get("loc")
        lines.append(
            f"- {_short(driver)}"
            + (f" inside {_short(always)}" if always else "")
            + (f" [{loc}]" if loc else "")
        )
    return "\n".join(lines)


def _distill_receivers(data: dict) -> str:
    if data.get("not_found"):
        return f"(signal not found: {data.get('detail')})"
    receivers = data.get("receivers", [])
    if not receivers:
        return "(no readers found)"
    shown = receivers[:_MAX_NODES_IN_SUMMARY]
    lines = [f"- {_short(r.get('reader', {}))}" for r in shown]
    text = "\n".join(lines)
    if len(receivers) > _MAX_NODES_IN_SUMMARY:
        text += f"\n... and {len(receivers) - _MAX_NODES_IN_SUMMARY} more (truncated)"
    return text


class SearchArgs(BaseModel):
    query: str = Field(description="Signal, register, module, or port name (or substring) to search for")


class SignalModuleArgs(BaseModel):
    signal: str = Field(description="Signal or register name")
    module: str | None = Field(default=None, description="Owning module name, if known")


class FaninFanoutArgs(SignalModuleArgs):
    max_depth: int = Field(default=4, description="Maximum BFS hops to traverse")


class PathArgs(BaseModel):
    source: str = Field(description="Source signal name")
    destination: str = Field(description="Destination signal name")
    module: str | None = Field(default=None)


class ModuleArgs(BaseModel):
    module: str = Field(description="Module name")


def build_graph_tools(cfg: GraphDbConfig) -> list[StructuredTool]:
    client = GraphDBClient(cfg)

    def search(query: str) -> str:
        """Search the design graph for a signal, register, module, or port by name."""
        return _distill_search(client.get("/search", {"q": query}))

    def trace_driver(signal: str, module: str | None = None) -> str:
        """Find what drives (assigns) a given signal — the always-block/assignment/expression, with file:line."""
        return _distill_driver(client.get("/driver", {"signal": signal, "module": module}))

    def trace_receivers(signal: str, module: str | None = None) -> str:
        """Find what reads a given signal downstream."""
        return _distill_receivers(client.get("/receivers", {"signal": signal, "module": module}))

    def fanin(signal: str, module: str | None = None, max_depth: int = 4) -> str:
        """Compute the fanin cone (everything this signal transitively depends on) up to max_depth hops."""
        data = client.get("/fanin", {"signal": signal, "module": module, "max_depth": max_depth})
        return _distill_fanin_fanout(data, "fanin")

    def fanout(signal: str, module: str | None = None, max_depth: int = 4) -> str:
        """Compute the fanout cone (everything that transitively depends on this signal) up to max_depth hops."""
        data = client.get("/fanout", {"signal": signal, "module": module, "max_depth": max_depth})
        return _distill_fanin_fanout(data, "fanout")

    def dependency_path(source: str, destination: str, module: str | None = None) -> str:
        """Find the shortest dependency path between two signals, if one exists."""
        data = client.get("/path", {"source": source, "destination": destination, "module": module})
        if data.get("not_found"):
            return f"(not found: {data.get('detail')})"
        if not data.get("found"):
            return "(no dependency path exists between these two signals)"
        return " -> ".join(_short(n) for n in data.get("path", []))

    def clock_domain(signal: str, module: str | None = None) -> str:
        """Find the clock domain a signal/register belongs to."""
        data = client.get("/signal/clock-domain", {"name": signal, "module": module})
        return json.dumps(data)[:_MAX_JSON_CHARS]

    def reset_tree(signal: str, module: str | None = None) -> str:
        """Find the reset domain / reset tree a signal/register belongs to."""
        data = client.get("/signal/reset-tree", {"name": signal, "module": module})
        return json.dumps(data)[:_MAX_JSON_CHARS]

    def module_hierarchy(module: str) -> str:
        """Get the instance hierarchy under a given module."""
        data = client.get("/module/hierarchy", {"name": module})
        return json.dumps(data)[:_MAX_JSON_CHARS]

    return [
        StructuredTool.from_function(search, name="graph_search", args_schema=SearchArgs),
        StructuredTool.from_function(trace_driver, name="graph_trace_driver", args_schema=SignalModuleArgs),
        StructuredTool.from_function(trace_receivers, name="graph_trace_receivers", args_schema=SignalModuleArgs),
        StructuredTool.from_function(fanin, name="graph_fanin", args_schema=FaninFanoutArgs),
        StructuredTool.from_function(fanout, name="graph_fanout", args_schema=FaninFanoutArgs),
        StructuredTool.from_function(dependency_path, name="graph_dependency_path", args_schema=PathArgs),
        StructuredTool.from_function(clock_domain, name="graph_clock_domain", args_schema=SignalModuleArgs),
        StructuredTool.from_function(reset_tree, name="graph_reset_tree", args_schema=SignalModuleArgs),
        StructuredTool.from_function(module_hierarchy, name="graph_module_hierarchy", args_schema=ModuleArgs),
    ]
