"""Structural health of a graph DB, independent of any debug run: node/edge
counts by type, orphan data-nodes (a Signal/Register/Port with no edges at
all — unreachable by any traversal), and dangling edges (referencing a node
id that doesn't exist, which silently breaks BFS-based fanin/fanout).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass
class StructuralMetrics:
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    total_nodes: int
    total_edges: int
    orphan_node_count: int
    orphan_node_examples: list[str] = field(default_factory=list)
    dangling_edge_count: int = 0

    @property
    def orphan_ratio(self) -> float:
        data_nodes = sum(
            v for k, v in self.node_counts.items() if k in ("Signal", "Register", "Port")
        )
        return self.orphan_node_count / data_nodes if data_nodes else 0.0


def compute(db_path: str) -> StructuralMetrics:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        node_counts = dict(cur.execute("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type").fetchall())
        edge_counts = dict(cur.execute("SELECT rel, COUNT(*) FROM edges GROUP BY rel").fetchall())

        cur.execute(
            """
            SELECT id FROM nodes
            WHERE node_type IN ('Signal', 'Register', 'Port')
              AND id NOT IN (SELECT src FROM edges)
              AND id NOT IN (SELECT dst FROM edges)
            """
        )
        orphans = [r[0] for r in cur.fetchall()]

        cur.execute(
            """
            SELECT COUNT(*) FROM edges
            WHERE src NOT IN (SELECT id FROM nodes) OR dst NOT IN (SELECT id FROM nodes)
            """
        )
        dangling = cur.fetchone()[0]

        return StructuralMetrics(
            node_counts=node_counts,
            edge_counts=edge_counts,
            total_nodes=sum(node_counts.values()),
            total_edges=sum(edge_counts.values()),
            orphan_node_count=len(orphans),
            orphan_node_examples=orphans[:10],
            dangling_edge_count=dangling,
        )
    finally:
        conn.close()
