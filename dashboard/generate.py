"""Builds the reviewer-facing static dashboard from everything under
data/runs/* (+ an optional graph_eval report), and writes a single
self-contained docs/index.html for GitHub Pages.

CLI: python -m dashboard.generate --runs-root data/runs --graph-eval data/graph_eval_report.json --out docs/index.html
"""
from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunView:
    run_id: str
    suite: str = "unknown"
    test: str = "(unknown)"
    phase: str = "unknown"
    fail_signature: str = ""
    error_class: str = ""
    file: str | None = None
    line: int | None = None
    explanation: str = ""
    confidence_score: float | None = None
    confidence_label: str | None = None
    confidence_rationale: str = ""
    evidence: list[dict] = None
    patch_diff: str = ""
    branch: str | None = None
    rerun_passed: bool | None = None
    rerun_summary: str = ""
    grade_verdict: str | None = None
    grade_expected: str | None = None
    tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_run(run_dir: Path) -> RunView:
    view = RunView(run_id=run_dir.name)
    events = _load_jsonl(run_dir / "trail.jsonl")

    for e in events:
        if e["event"] == "awaiting_approval":
            view.test = e.get("test", view.test)
            view.suite = e.get("suite", view.suite)
            view.fail_signature = e.get("fail_signature") or view.fail_signature
            view.error_class = e.get("error_class") or view.error_class
            view.file = e.get("file")
            view.line = e.get("line")
            view.explanation = e.get("explanation", "")
            view.confidence_score = e.get("confidence_score")
            view.confidence_label = e.get("confidence_label")
            view.confidence_rationale = e.get("confidence_rationale", "")
            view.evidence = e.get("evidence", [])
        if e["event"] == "run_finished":
            view.phase = e.get("phase", view.phase)
            view.branch = e.get("branch")
            view.rerun_passed = e.get("rerun_passed")
            view.rerun_summary = e.get("summary", "") or ""

    patch_path = run_dir / "patch.diff"
    if patch_path.is_file():
        view.patch_diff = patch_path.read_text()

    grade_path = run_dir / "grade.json"
    if grade_path.is_file():
        grade = json.loads(grade_path.read_text())
        view.grade_verdict = grade.get("verdict")
        if grade.get("expected_file"):
            view.grade_expected = f"bug #{grade.get('expected_bug_number')} — {grade['expected_file']}:{grade.get('expected_lines')}"

    token_path = run_dir / "token_report.json"
    if token_path.is_file():
        tr = json.loads(token_path.read_text())
        view.tokens = tr.get("total_tokens", 0)
        view.cost_usd = tr.get("estimated_cost_usd", 0.0)

    if view.phase == "unknown" and events:
        view.phase = "awaiting_approval" if any(e["event"] == "awaiting_approval" for e in events) else "unknown"

    return view


def load_all_runs(runs_root: Path) -> list[RunView]:
    if not runs_root.is_dir():
        return []
    runs = [load_run(d) for d in sorted(runs_root.iterdir()) if d.is_dir()]
    return sorted(runs, key=lambda r: r.test)


_STATUS = {
    "verified": ("good", "Verified"),
    "failed": ("critical", "Fix failed"),
    "rejected": ("warning", "Rejected"),
    "awaiting_approval": ("warning", "Awaiting approval"),
    "unknown": ("warning", "Incomplete"),
}

_GRADE_STATUS = {
    "correct": ("good", "Correct"),
    "file_only": ("warning", "File match, line off"),
    "incorrect": ("critical", "Incorrect"),
    "no_matching_bug_entry": ("warning", "No answer-key entry"),
}


def _badge(status_key: str, table: dict) -> str:
    role, label = table.get(status_key, ("warning", status_key))
    return f'<span class="badge badge-{role}">{html.escape(label)}</span>'


def _confidence_bar(score: float | None, label: str | None) -> str:
    if score is None:
        return '<span class="muted">n/a</span>'
    role = {"high": "good", "medium": "warning", "low": "critical"}.get(label or "", "warning")
    pct = round(score * 100)
    return f"""<div class="conf-wrap" title="{pct}% ({html.escape(label or '')})">
  <div class="conf-track"><div class="conf-fill conf-{role}" style="width:{pct}%"></div></div>
  <span class="conf-label">{pct}%</span>
</div>"""


_TOOL_LABELS = {
    "graph_search": "Searched the design graph",
    "graph_trace_driver": "Traced the signal's driver",
    "graph_trace_receivers": "Traced the signal's readers",
    "graph_fanin": "Computed the fanin cone (upstream dependencies)",
    "graph_fanout": "Computed the fanout cone (downstream dependents)",
    "graph_dependency_path": "Checked the dependency path between two signals",
    "graph_clock_domain": "Checked the clock domain",
    "graph_reset_tree": "Checked the reset domain",
    "graph_module_hierarchy": "Inspected the module instance hierarchy",
    "spec_search": "Searched the design spec for expected behavior",
    "rtl_read_file": "Read the RTL source",
}


def _step_label(step: dict) -> str:
    label = _TOOL_LABELS.get(step["tool"], step["tool"])
    key_arg = step["args"].get("signal") or step["args"].get("query") or step["args"].get("path") or step["args"].get("module")
    return f"{label}" + (f" (<code>{html.escape(str(key_arg))}</code>)" if key_arg else "")


def _investigation_plan(evidence: list[dict]) -> str:
    if not evidence:
        return ""
    items = "".join(f"<li>{_step_label(step)}</li>" for step in evidence)
    return f"""
    <div class="kv-block">
      <span class="k">Investigation plan</span>
      <ol class="plan">{items}</ol>
    </div>"""


def _evidence_steps(evidence: list[dict]) -> str:
    if not evidence:
        return ""
    rows = "".join(
        f"""<li>
          <div class="step-head">Step {i + 1}: {_step_label(step)}</div>
          <pre class="step-result">{html.escape(step.get("summary", ""))}</pre>
        </li>"""
        for i, step in enumerate(evidence)
    )
    return f"""
    <details class="evidence">
      <summary>Retrieved evidence ({len(evidence)} step{'s' if len(evidence) != 1 else ''})</summary>
      <ol class="evidence-list">{rows}</ol>
    </details>"""


def _run_card(r: RunView) -> str:
    grade_html = ""
    if r.grade_verdict:
        grade_html = f"""
        <div class="kv"><span class="k">Answer-key check</span><span class="v">{_badge(r.grade_verdict, _GRADE_STATUS)}
        {f'<span class="muted"> expected {html.escape(r.grade_expected)}</span>' if r.grade_expected else ''}</span></div>"""

    patch_html = ""
    if r.patch_diff:
        patch_html = f"""
        <details class="patch">
          <summary>Proposed patch</summary>
          <pre class="diff">{html.escape(r.patch_diff)}</pre>
        </details>"""

    rerun_html = ""
    if r.rerun_passed is not None:
        rerun_role = "good" if r.rerun_passed else "critical"
        rerun_html = f"""
        <div class="kv"><span class="k">Rerun</span><span class="v">
          <span class="badge badge-{rerun_role}">{'PASS' if r.rerun_passed else 'FAIL'}</span>
          <span class="muted">{html.escape(r.branch or '')}</span></span></div>"""

    fail_sig_html = ""
    if r.fail_signature:
        fail_sig_html = f"""
        <div class="kv-block">
          <span class="k">Fail signature{f' ({html.escape(r.error_class)})' if r.error_class else ''}</span>
          <pre class="fail-sig">{html.escape(r.fail_signature)}</pre>
        </div>"""

    return f"""
    <article class="card">
      <header class="card-head">
        <div>
          <div class="breadcrumb">{html.escape(r.suite)} regression</div>
          <h3>{html.escape(r.test)}</h3>
        </div>
        {_badge(r.phase, _STATUS)}
      </header>
      {fail_sig_html}
      {_investigation_plan(r.evidence)}
      <div class="kv"><span class="k">Hypothesis / root cause</span><span class="v"><code>{html.escape(r.file or '?')}:{r.line if r.line else '?'}</code></span></div>
      <div class="kv"><span class="k">Confidence</span><span class="v">{_confidence_bar(r.confidence_score, r.confidence_label)}</span></div>
      <p class="explanation">{html.escape(r.explanation)}</p>
      {_evidence_steps(r.evidence)}
      {grade_html}
      {rerun_html}
      <div class="kv"><span class="k">Tokens</span><span class="v muted">{r.tokens:,} (~${r.cost_usd:.4f})</span></div>
      {patch_html}
    </article>"""


def _summary_stats(runs: list[RunView]) -> dict:
    verified = sum(1 for r in runs if r.phase == "verified")
    correct = sum(1 for r in runs if r.grade_verdict == "correct")
    graded = sum(1 for r in runs if r.grade_verdict is not None)
    total_tokens = sum(r.tokens for r in runs)
    total_cost = sum(r.cost_usd for r in runs)
    avg_conf = (
        sum(r.confidence_score for r in runs if r.confidence_score is not None)
        / max(1, sum(1 for r in runs if r.confidence_score is not None))
    )
    return {
        "total": len(runs),
        "verified": verified,
        "correct": correct,
        "graded": graded,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_confidence": avg_conf,
    }


def _graph_eval_panel(graph_eval_path: Path | None) -> str:
    if not graph_eval_path or not graph_eval_path.is_file():
        return ""
    d = json.loads(graph_eval_path.read_text())
    return f"""
    <section class="panel">
      <h2>Graph DB health</h2>
      <div class="stat-row">
        <div class="stat"><div class="stat-num">{d['score']}</div><div class="stat-label">Overall / 100</div></div>
        <div class="stat"><div class="stat-num">{d['structural_score']}</div><div class="stat-label">Structural</div></div>
        <div class="stat"><div class="stat-num">{d['retrieval_score']}</div><div class="stat-label">Retrieval</div></div>
        <div class="stat"><div class="stat-num">{d['coverage_score']}</div><div class="stat-label">Coverage</div></div>
      </div>
      <p class="muted">{d['structural']['total_nodes']:,} nodes / {d['structural']['total_edges']:,} edges &middot;
      {d['structural']['orphan_node_count']} orphan nodes &middot; {d['structural']['dangling_edge_count']} dangling edges</p>
    </section>"""


_CSS = """
:root {
  color-scheme: light;
  --surface-0: #fcfcfb; --surface-1: #ffffff; --border: #e5e4e0;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #83817a;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2b;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8d84;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #e66767;
    --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2b;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8d84;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #e66767;
  --accent: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }
h1 { font-size: 1.7rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.1rem; margin: 0 0 1rem; }
.subtitle { color: var(--text-secondary); margin: 0 0 2rem; max-width: 60ch; }
.stat-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
.stat { min-width: 110px; }
.stat-num { font-size: 1.8rem; font-weight: 700; }
.stat-label { font-size: 0.8rem; color: var(--text-secondary); }
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.25rem 1.5rem; margin-bottom: 2rem; }
.grid { display: grid; grid-template-columns: 1fr; gap: 1rem; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1.5rem; }
.card-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.75rem; }
.card-head h3 { margin: 0; font-size: 1.05rem; }
.badge { font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 999px;
  border: 1px solid transparent; white-space: nowrap; }
.badge-good { color: var(--good); border-color: var(--good); }
.badge-warning { color: #8a5a00; border-color: var(--warning); background: color-mix(in srgb, var(--warning) 15%, transparent); }
:root[data-theme="dark"] .badge-warning, @media (prefers-color-scheme: dark) { .badge-warning { color: var(--warning); } }
.badge-critical { color: var(--critical); border-color: var(--critical); }
.badge-serious { color: var(--serious); border-color: var(--serious); }
.kv { display: flex; gap: 0.5rem; padding: 0.2rem 0; font-size: 0.9rem; align-items: center; flex-wrap: wrap; }
.kv .k { color: var(--text-secondary); min-width: 130px; }
.kv .v { color: var(--text-primary); }
code { background: var(--surface-0); border: 1px solid var(--border); border-radius: 4px;
  padding: 0.1rem 0.35rem; font-size: 0.85rem; }
.muted { color: var(--text-muted); font-size: 0.85rem; }
.explanation { font-size: 0.9rem; color: var(--text-secondary); margin: 0.5rem 0; }
.conf-wrap { display: flex; align-items: center; gap: 0.5rem; width: 160px; }
.conf-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 3px; }
.conf-good { background: var(--good); }
.conf-warning { background: var(--warning); }
.conf-critical { background: var(--critical); }
.conf-label { font-size: 0.78rem; color: var(--text-secondary); }
details.patch, details.evidence { margin-top: 0.75rem; }
details.patch summary, details.evidence summary { cursor: pointer; font-size: 0.85rem; color: var(--accent); }
pre.diff { background: var(--surface-0); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.75rem 1rem; overflow-x: auto; font-size: 0.8rem; margin-top: 0.5rem; }
.breadcrumb { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.03em; }
.kv-block { margin: 0.6rem 0; }
.kv-block .k { color: var(--text-secondary); font-size: 0.85rem; display: block; margin-bottom: 0.3rem; }
pre.fail-sig { background: var(--surface-0); border: 1px solid var(--border); border-left: 3px solid var(--serious);
  border-radius: 6px; padding: 0.6rem 0.85rem; font-size: 0.82rem; overflow-x: auto; white-space: pre-wrap; margin: 0; }
ol.plan { margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: var(--text-secondary); }
ol.plan li { margin-bottom: 0.15rem; }
ol.plan code { font-size: 0.82rem; }
ol.evidence-list { list-style: none; margin: 0.5rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 0.6rem; }
ol.evidence-list li { border-left: 2px solid var(--border); padding-left: 0.75rem; }
.step-head { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.2rem; }
pre.step-result { background: var(--surface-0); border: 1px solid var(--border); border-radius: 6px;
  padding: 0.5rem 0.7rem; font-size: 0.78rem; overflow-x: auto; white-space: pre-wrap; margin: 0; max-height: 220px; overflow-y: auto; }
footer { margin-top: 2.5rem; font-size: 0.8rem; color: var(--text-muted); }
"""


def render(runs: list[RunView], graph_eval_path: Path | None) -> str:
    stats = _summary_stats(runs)
    cards = "\n".join(_run_card(r) for r in runs) or '<p class="muted">No runs yet.</p>'
    graph_panel = _graph_eval_panel(graph_eval_path)

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RTL Debug Agent — Dashboard</title>
<style>{_CSS}</style>
</head><body><div class="wrap">
<h1>🐛 RTL Debug Agent</h1>
<p class="subtitle">Graph-grounded RTL debugging: each run below diagnosed a failing simulation
test using only a deterministic design graph, the spec, and RTL source — never the project's
answer key — then proposed a patch, waited for human approval, and reran the test to verify.</p>

<section class="panel">
  <h2>Summary</h2>
  <div class="stat-row">
    <div class="stat"><div class="stat-num">{stats['total']}</div><div class="stat-label">Runs</div></div>
    <div class="stat"><div class="stat-num">{stats['verified']}</div><div class="stat-label">Fixes verified</div></div>
    <div class="stat"><div class="stat-num">{stats['correct']}/{stats['graded']}</div><div class="stat-label">Matched answer key</div></div>
    <div class="stat"><div class="stat-num">{stats['avg_confidence']*100:.0f}%</div><div class="stat-label">Avg. confidence</div></div>
    <div class="stat"><div class="stat-num">${stats['total_cost']:.3f}</div><div class="stat-label">Total LLM cost</div></div>
  </div>
</section>

{graph_panel}

<h2>Runs</h2>
<div class="grid">
{cards}
</div>

<footer>Generated by dashboard/generate.py. Diagnosis loop scoped away from
docs/verification/bug_list.md by construction (see agent/tools/rtl_tools.py and
spec_tools.py); grading against it happens only after a diagnosis is final.</footer>
</div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="data/runs")
    parser.add_argument("--graph-eval", default="data/graph_eval_report.json")
    parser.add_argument("--out", default="docs/index.html")
    args = parser.parse_args()

    runs = load_all_runs(Path(args.runs_root))
    graph_eval_path = Path(args.graph_eval)
    html_out = render(runs, graph_eval_path if graph_eval_path.is_file() else None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out)
    print(f"Dashboard written to {out_path} ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
