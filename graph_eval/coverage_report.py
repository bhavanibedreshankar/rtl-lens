"""Cross-checks the graph against the RTL source it claims to represent, to
catch silent gaps — e.g. a file with `always` blocks the elaborator didn't
turn into `AlwaysBlock` nodes. Coarse (regex-based) by design: this is a
sanity check on the graph DB itself, not a substitute for the elaborator.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_ALWAYS_RE = re.compile(r"\balways(_ff|_comb|_latch)?\s*[@(]")
_MODULE_RE = re.compile(r"\bmodule\s+(\w+)")


@dataclass
class FileCoverage:
    file: str
    always_in_source: int
    always_blocks_in_graph: int

    @property
    def ratio(self) -> float:
        """Clipped to 1.0: the graph holds one AlwaysBlock per *elaborated
        instance*, not per source declaration, so a heavily-instantiated leaf
        module (e.g. a systolic array's PE, instantiated hundreds of times)
        legitimately has far more graph nodes than source `always` keywords.
        That's expected over-representation, not a signal worth scoring on —
        only under-representation (a real gap) should move the score."""
        if not self.always_in_source:
            return 1.0
        return min(1.0, self.always_blocks_in_graph / self.always_in_source)


@dataclass
class CoverageReport:
    files: list[FileCoverage] = field(default_factory=list)

    @property
    def mean_ratio(self) -> float:
        return sum(f.ratio for f in self.files) / len(self.files) if self.files else 1.0

    @property
    def under_covered(self) -> list[FileCoverage]:
        return [f for f in self.files if f.ratio < 0.8]


def compute(db_path: str, rtl_root: Path) -> CoverageReport:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT attrs FROM nodes WHERE node_type = 'AlwaysBlock'"
        ).fetchall()
    finally:
        conn.close()

    import json

    graph_counts: dict[str, int] = {}
    for (attrs_json,) in rows:
        try:
            attrs = json.loads(attrs_json) if attrs_json else {}
        except json.JSONDecodeError:
            continue
        loc = attrs.get("loc", "")
        file_part = loc.split(":")[0] if loc else None
        if file_part:
            graph_counts[file_part] = graph_counts.get(file_part, 0) + 1

    report = CoverageReport()
    for sv_file in sorted(rtl_root.rglob("*.sv")):
        rel = str(sv_file.relative_to(rtl_root.parent))
        text = sv_file.read_text(errors="ignore")
        always_count = len(_ALWAYS_RE.findall(text))
        if always_count == 0:
            continue
        report.files.append(
            FileCoverage(file=rel, always_in_source=always_count, always_blocks_in_graph=graph_counts.get(rel, 0))
        )
    return report
