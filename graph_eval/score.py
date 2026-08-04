"""Rolls structural health, retrieval accuracy, and source coverage into one
weighted 0-100 "graph health score" plus a written report — this is what a
debug run can point back to ("ran against a graph DB scoring N/100") so a
reviewer can judge how much to trust a given diagnosis.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from config.schema import ProjectConfig
from graph_eval import coverage_report, retrieval_benchmark, structural_metrics

_WEIGHTS = {"structural": 0.3, "retrieval": 0.5, "coverage": 0.2}


@dataclass
class GraphHealthReport:
    score: float
    structural_score: float
    retrieval_score: float
    coverage_score: float
    structural: dict
    retrieval: dict
    coverage: dict


def _resolve_db_path(config: ProjectConfig) -> Path:
    """The graph DB's own SQLite cache lives alongside RTL_Graph's checked-out
    source, referenced only by `graph_db.local_fallback_cwd` in config — this
    eval reads the cache file directly, since structural/coverage checks need
    raw SQL the REST API doesn't expose."""
    cwd = config.graph_db.local_fallback_cwd
    if cwd is None:
        raise RuntimeError("graph_db.local_fallback_cwd must be set in config to run graph_eval")
    db_path = cwd / "data" / "rtlgraph.db"
    if not db_path.is_file():
        raise RuntimeError(f"no graph DB cache found at {db_path} — build it first (see RTL_Graph's README)")
    return db_path


def run(config: ProjectConfig) -> GraphHealthReport:
    db_path = _resolve_db_path(config)

    structural = structural_metrics.compute(str(db_path))
    retrieval = retrieval_benchmark.run(config.graph_db)
    coverage = coverage_report.compute(str(db_path), config.rtl_repo.rtl_root)

    structural_score = max(0.0, 100 * (1 - structural.orphan_ratio) - (10 if structural.dangling_edge_count else 0))
    retrieval_score = 100 * retrieval.hit_rate
    coverage_score = 100 * coverage.mean_ratio

    overall = (
        _WEIGHTS["structural"] * structural_score
        + _WEIGHTS["retrieval"] * retrieval_score
        + _WEIGHTS["coverage"] * coverage_score
    )

    return GraphHealthReport(
        score=round(overall, 1),
        structural_score=round(structural_score, 1),
        retrieval_score=round(retrieval_score, 1),
        coverage_score=round(coverage_score, 1),
        structural=asdict(structural),
        retrieval={
            "hit_rate": retrieval.hit_rate,
            "mean_latency_ms": round(retrieval.mean_latency_ms, 1),
            "results": [asdict(r) for r in retrieval.results],
        },
        coverage={
            "mean_ratio": coverage.mean_ratio,
            "under_covered": [asdict(f) for f in coverage.under_covered],
            "files_checked": len(coverage.files),
        },
    )


def write_json(report: GraphHealthReport, path: Path) -> None:
    path.write_text(json.dumps(asdict(report), indent=2))


def render_html(report: GraphHealthReport) -> str:
    def row(label: str, value: str) -> str:
        return f"<tr><td>{label}</td><td>{value}</td></tr>"

    under_covered_rows = "".join(
        f"<tr><td>{f['file']}</td><td>{f['always_blocks_in_graph']}/{f['always_in_source']}</td></tr>"
        for f in report.coverage["under_covered"]
    ) or "<tr><td colspan=2>none</td></tr>"

    retrieval_rows = "".join(
        f"<tr><td>{r['name']}</td><td>{'PASS' if r['passed'] else 'FAIL'}</td>"
        f"<td>{r['latency_ms']:.1f} ms</td></tr>"
        for r in report.retrieval["results"]
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Graph DB Health Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; color: #1a1a1a; }}
h1 {{ font-size: 1.6rem; }}
.score {{ font-size: 3rem; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
th {{ background: #f5f5f5; }}
</style></head><body>
<h1>Graph DB Health Report</h1>
<div class="score">{report.score}/100</div>
<table>
{row("Structural score", f"{report.structural_score}/100")}
{row("Retrieval score", f"{report.retrieval_score}/100")}
{row("Coverage score", f"{report.coverage_score}/100")}
{row("Total nodes", report.structural['total_nodes'])}
{row("Total edges", report.structural['total_edges'])}
{row("Orphan data-nodes", report.structural['orphan_node_count'])}
{row("Dangling edges", report.structural['dangling_edge_count'])}
</table>
<h2>Retrieval benchmark</h2>
<table><tr><th>Check</th><th>Result</th><th>Latency</th></tr>{retrieval_rows}</table>
<h2>Under-covered source files (&lt;80% of always-blocks represented)</h2>
<table><tr><th>File</th><th>AlwaysBlock nodes / always in source</th></tr>{under_covered_rows}</table>
</body></html>"""
