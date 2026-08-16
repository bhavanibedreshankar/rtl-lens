"""Precision/recall/latency of the graph DB's retrieval API against a small
set of known-answer queries. These facts were independently hand-verified
against the actual RTL source (not pulled from bug_list.md — this eval is
about the graph's general query quality, not any specific debug session), so
extending the benchmark for a new design just means adding more entries here
after checking the RTL by hand once.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.tools.graph_tools import GraphDBClient
from config.schema import GraphDbConfig

# (query_type, args, assertion) — assertion receives the raw JSON response.
BENCHMARK: list[dict] = [
    {
        "name": "search finds a known register",
        "endpoint": "/search",
        "params": {"q": "irq_status_q"},
        "check": lambda r: any(m.get("id") == "register:tpe_cmd_proc.irq_status_q" for m in r.get("results", [])),
    },
    {
        "name": "driver trace resolves to the writing always-block's file",
        "endpoint": "/driver",
        "params": {"signal": "irq_status_q", "module": "tpe_cmd_proc"},
        "check": lambda r: any(
            "tpe_cmd_proc.sv" in (d.get("always_block") or d.get("driver") or {}).get("loc", "")
            for d in r.get("drivers", [])
        ),
    },
    {
        "name": "clock domain resolves to a real clock",
        "endpoint": "/signal/clock-domain",
        "params": {"name": "irq_status_q", "module": "tpe_cmd_proc"},
        "check": lambda r: any(c.get("name") == "clk" for c in r.get("clock_domains", [])),
    },
    {
        # Companion to the clock-domain check above, on the same register:
        # RTL_Graph had a real bug where clock/reset tie-breaking depended on
        # Python's randomized set/hash iteration order, so a design could get
        # clk and rst_n flipped between runs (fixed in RTL_Graph commit
        # 1496fd9). The clock check alone would already catch a swap on this
        # signal, but asserting the reset side too makes the benchmark
        # symmetric and catches reset-specific regressions (wrong edge,
        # wrong domain node) that a clock-only check can't see.
        "name": "reset domain resolves to the real reset, not the clock",
        "endpoint": "/signal/reset-tree",
        "params": {"name": "irq_status_q", "module": "tpe_cmd_proc"},
        "check": lambda r: any(c.get("name") == "rst_n" for c in r.get("reset_domains", [])),
    },
    {
        "name": "fanin of a register is non-empty",
        "endpoint": "/fanin",
        "params": {"signal": "irq_status_q", "module": "tpe_cmd_proc", "max_depth": 3},
        "check": lambda r: len(r.get("fanin", [])) > 0,
    },
    {
        "name": "module hierarchy resolves the top module",
        "endpoint": "/module/hierarchy",
        "params": {"name": "tpe_top"},
        "check": lambda r: "module" in r,
    },
]


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    latency_ms: float
    error: str | None = None


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0.0


def run(graph_db_cfg: GraphDbConfig) -> BenchmarkReport:
    client = GraphDBClient(graph_db_cfg)
    report = BenchmarkReport()
    for case in BENCHMARK:
        start = time.monotonic()
        try:
            data = client.get(case["endpoint"], case["params"])
            passed = bool(case["check"](data))
            error = None
        except Exception as exc:  # noqa: BLE001
            passed = False
            error = str(exc)
        latency_ms = (time.monotonic() - start) * 1000
        report.results.append(BenchmarkResult(name=case["name"], passed=passed, latency_ms=latency_ms, error=error))
    return report
