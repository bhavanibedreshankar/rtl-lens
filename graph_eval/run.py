"""CLI: python -m graph_eval.run --config config/tpe.yaml --report docs/graph_eval_report.html"""
from __future__ import annotations

import argparse
from pathlib import Path

from config.schema import ProjectConfig
from graph_eval.score import render_html, run, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", default="docs/graph_eval_report.html")
    parser.add_argument("--json", default=None, help="Optional path to also write raw JSON")
    args = parser.parse_args()

    config = ProjectConfig.load(args.config)
    report = run(config)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_html(report))

    if args.json:
        write_json(report, Path(args.json))

    print(f"Graph health score: {report.score}/100")
    print(f"  structural: {report.structural_score}/100")
    print(f"  retrieval:  {report.retrieval_score}/100")
    print(f"  coverage:   {report.coverage_score}/100")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
