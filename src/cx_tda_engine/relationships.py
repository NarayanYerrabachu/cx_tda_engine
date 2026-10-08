"""Label co-occurrence between two grouping columns (the Graph tab's relationships)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .config import TDAConfig, resolve

_MISSING = ("nan", "None", "Unknown", "N/A", "")


def compute_relationships(
    df: pd.DataFrame,
    record_ids: list[str],
    cluster_labels: np.ndarray,
    group_cols: list[str],
    combined_scores: np.ndarray | None = None,
    suspicious_ids: set[str] | None = None,
    basis_mask: np.ndarray | None = None,
    relationship_type: str = "cooccurrence",
    config: TDAConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Relationships from the co-occurrence of ``group_cols[0]`` x ``group_cols[1]``.

    Every row of the basis (all rows, or those where ``basis_mask`` is True,
    e.g. "projects with cost overrun > 50 %") contributes one label pair; a
    pair becomes a relationship when it occurs at least twice. Weight is the
    row count, score is weight / n_rows. Up to 60 relationships are returned,
    sorted by weight; the 30 heaviest also form ``graph_edges``.

    DBSCAN cluster labels only contribute members to ``graph_nodes``; clusters
    are never linked here. The Mapper graph is a separate structure.

    With ``combined_scores`` / ``suspicious_ids`` every relationship also
    carries full-population counts (never derived from the capped ``sources``
    samples): ``extra.n_anomalous``, ``extra.n_suspicious`` and per-side
    ``a_stats`` / ``b_stats`` (n_records, n_links, n_anomalous, n_suspicious).

    Returns ``(relationships, graph_nodes, graph_edges)``; all empty when fewer
    than two of the ``group_cols`` exist in ``df``.
    """
    cfg = resolve(config)
    cols = [c for c in group_cols if c in df.columns][:2]
    if len(cols) < 2:
        return [], [], []
    col_a, col_b = cols

    n_rows = len(df)
    dfp = df.reset_index(drop=True)
    rid = np.asarray(record_ids, dtype=object)
    anom_flag = (np.asarray(combined_scores, dtype=float) >= cfg.anomaly_high
                 if combined_scores is not None and len(combined_scores) == n_rows else None)
    susp_flag = (np.array([str(r) in suspicious_ids for r in record_ids], dtype=bool)
                 if suspicious_ids is not None and len(record_ids) == n_rows else None)

    def _counts(pos: np.ndarray) -> dict[str, int]:
        out: dict[str, int] = {}
        if anom_flag is not None:
            out["n_anomalous"] = int(anom_flag[pos].sum())
        if susp_flag is not None:
            out["n_suspicious"] = int(susp_flag[pos].sum())
        return out

    basis = dfp if basis_mask is None else dfp[np.asarray(basis_mask, dtype=bool)]
    relationships: list[dict[str, Any]] = []
    nodes_map: dict[str, set[str]] = defaultdict(set)
    for (a_val, b_val), grp in basis.groupby([col_a, col_b]):
        a, b = str(a_val), str(b_val)
        if len(grp) < 2 or a in _MISSING or b in _MISSING:
            continue
        pos = grp.index.to_numpy()
        weight = len(grp)
        src_ids = [str(x) for x in rid[pos[:5]]] if len(rid) == n_rows else []
        nodes_map[a].update(src_ids)
        nodes_map[b].update(src_ids)
        relationships.append({
            "kind":    "relationship",
            "title":   f"{a} ↔ {b}",
            "score":   round(weight / (n_rows + 1e-9), 4),
            "sources": src_ids,
            "detail":  f"{weight} records share {a} / {b}.",
            "extra":   {"a": a, "b": b, "weight": weight, "type": relationship_type, **_counts(pos)},
        })

    # Cluster membership: the top group value of every cluster gets its members.
    if len(cluster_labels) == n_rows and len(rid) == n_rows:
        for cid in sorted(set(cluster_labels.tolist())):
            if cid == -1:
                continue
            pos = np.where(cluster_labels == cid)[0]
            top = dfp.iloc[pos][col_a].astype(str).value_counts().head(1)
            for val in top.index:
                nodes_map[str(val)].update(str(x) for x in rid[pos[:5]])

    relationships.sort(key=lambda r: r["extra"].get("weight", 0), reverse=True)

    if relationships:
        for side, col in (("a", col_a), ("b", col_b)):
            vals = basis[col].astype(str)
            links: dict[str, int] = defaultdict(int)
            for r in relationships:
                links[r["extra"][side]] += 1
            side_stats: dict[str, dict[str, int]] = {}
            for v, idx in vals.groupby(vals).groups.items():
                pos = np.asarray(idx)
                side_stats[str(v)] = {"n_records": int(len(pos)), "n_links": int(links.get(str(v), 0)),
                                      **_counts(pos)}
            for r in relationships:
                r["extra"][f"{side}_stats"] = side_stats.get(r["extra"][side])

    graph_nodes = [{"id": name, "docs": len(ids), "exposure": 0.0}
                   for name, ids in sorted(nodes_map.items(), key=lambda x: len(x[1]), reverse=True)[:30]]
    graph_edges = [{"a": r["extra"]["a"], "b": r["extra"]["b"], "w": r["extra"].get("weight", 1),
                    "sources": r["sources"]} for r in relationships[:30]]
    return relationships[:60], graph_nodes, graph_edges
