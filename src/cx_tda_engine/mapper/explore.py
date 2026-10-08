"""The exploration Mapper graph (:func:`run_mapper`): per-interval standardisation,
auto-tuned eps, risk tiers, neighbour lists, lens histogram and interval bands
for an interactive panel. Heavier than the pipeline graph, richer output.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

from ..config import TDAConfig, resolve
from ..core.risk import mapper_node_risk_level, mapper_risk_counts, node_risk_level
from .dbscan import AUTO_DEFAULT_EPS, auto_mapper_eps, dbscan_capped

log = logging.getLogger(__name__)


def _neighbors(graph: nx.Graph, nid: int, cfg: TDAConfig) -> list[dict[str, Any]]:
    """All neighbours of a Mapper node, most shared members first."""
    out = []
    for nb in graph.neighbors(nid):
        a = graph.nodes[nb]
        out.append({
            "id":         nb,
            "shared":     int(graph.edges[nid, nb].get("weight", 1)),
            "size":       a.get("size", 0),
            "anomaly":    a.get("avg_anomaly", 0.0),
            "risk_level": node_risk_level(a, cfg),
            "lift":       a.get("lift"),
        })
    out.sort(key=lambda x: x["shared"], reverse=True)
    return out


def run_mapper(
    X_norm: np.ndarray,
    lens_values: np.ndarray,
    n_intervals: int = 10,
    overlap: float = 0.4,
    dbscan_eps: float = AUTO_DEFAULT_EPS,
    dbscan_min_samples: int = 3,
    anomaly_scores: np.ndarray | None = None,
    record_ids: list[str] | None = None,
    config: TDAConfig | None = None,
) -> dict[str, Any]:
    """Exploration Mapper on ``X_norm`` with pre-computed lens values.

    Returns ``nodes`` (id, size, lens_center/low/high, interval_*, avg/max
    anomaly, n/frac anomalous, lift, risk_level, members, neighbors), ``edges``
    (source, target, weight), ``stats`` (counts, tiers, thresholds),
    ``lens_histogram`` and ``interval_bands``.

    Leaving ``dbscan_eps`` at its default auto-tunes it from the data.
    """
    cfg = resolve(config)
    n = len(X_norm)
    if n == 0:
        return {"nodes": [], "edges": [], "stats": {}, "lens_histogram": [], "interval_bands": []}
    scores = np.asarray(anomaly_scores, dtype=float) if anomaly_scores is not None else None
    has_scores = scores is not None and len(scores) == n
    base_rate = float(np.mean(scores >= cfg.anomaly_high)) if has_scores and scores is not None else 0.0

    lens = np.asarray(lens_values, dtype=float).ravel()
    l_min, l_max = float(lens.min()), float(lens.max())
    l_range = l_max - l_min or 1.0

    # Cover: width so that adjacent intervals overlap by `overlap`.
    width = l_range / (n_intervals - (n_intervals - 1) * overlap)
    step = width * (1.0 - overlap)
    interval_starts = [l_min + i * step for i in range(n_intervals)]

    if abs(dbscan_eps - AUTO_DEFAULT_EPS) < 1e-6:
        dbscan_eps = auto_mapper_eps(X_norm, dbscan_min_samples)

    graph = nx.Graph()
    node_id = 0
    for i, lo in enumerate(interval_starts):
        hi = lo + width
        mask = (lens >= lo - 1e-10) & (lens <= hi + 1e-10)
        idx = np.where(mask)[0]
        if len(idx) < dbscan_min_samples:
            continue
        X_sub = X_norm[idx]
        try:
            X_scaled = StandardScaler().fit_transform(X_sub)
        except Exception:
            X_scaled = X_sub

        labels = dbscan_capped(X_scaled, dbscan_eps, dbscan_min_samples, cfg.seed + i, cfg)

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            members_local = idx[labels == cluster_id].tolist()
            if not members_local:
                continue
            lens_vals_node = lens[members_local]
            avg_anom = max_anom = frac_anom = 0.0
            n_anom = 0
            if has_scores and scores is not None:
                ms = scores[members_local]
                avg_anom  = float(np.mean(ms))
                max_anom  = float(np.max(ms))
                n_anom    = int((ms >= cfg.anomaly_high).sum())
                frac_anom = n_anom / len(members_local)
            member_ids = ([record_ids[m] for m in members_local if m < len(record_ids)]
                          if record_ids else members_local)
            graph.add_node(node_id, **{
                "size":           len(members_local),
                "lens_center":    float(lens_vals_node.mean()),
                "lens_low":       float(lens_vals_node.min()),
                "lens_high":      float(lens_vals_node.max()),
                "interval_idx":   i,
                "interval_lo":    float(lo),
                "interval_hi":    float(hi),
                "avg_anomaly":    round(avg_anom, 4),
                "max_anomaly":    round(max_anom, 4),
                "n_anomalous":    n_anom,
                "frac_anomalous": round(frac_anom, 4),
                "lift":           round(frac_anom / base_rate, 3) if base_rate else None,
                "risk_level":     mapper_node_risk_level(frac_anom, base_rate, cfg),
                "members":        member_ids,
                "n_members":      len(member_ids),
            })
            node_id += 1

    # Edges: nodes can only share members if their intervals overlap, i.e. the
    # interval indices differ by less than width/step = 1/(1-overlap).
    node_list = list(graph.nodes(data=True))
    member_sets = {nid: set(attrs["members"]) for nid, attrs in node_list}
    max_gap = int(np.ceil(1.0 / max(1e-9, 1.0 - overlap)))
    for a in range(len(node_list)):
        nid_i, attrs_i = node_list[a]
        for b in range(a + 1, len(node_list)):
            nid_j, attrs_j = node_list[b]
            if abs(attrs_i["interval_idx"] - attrs_j["interval_idx"]) > max_gap:
                continue
            shared = member_sets[nid_i] & member_sets[nid_j]
            if shared:
                graph.add_edge(nid_i, nid_j, weight=len(shared))

    nodes_out = [{"id": nid, **attrs, "neighbors": _neighbors(graph, nid, cfg)}
                 for nid, attrs in graph.nodes(data=True)]
    edges_out = [{"source": u, "target": v, "weight": d.get("weight", 1)}
                 for u, v, d in graph.edges(data=True)]

    n_nodes   = graph.number_of_nodes()
    all_sizes = [attrs["size"] for _, attrs in graph.nodes(data=True)]
    stats = {
        "n_nodes":           n_nodes,
        "n_edges":           graph.number_of_edges(),
        "n_components":      nx.number_connected_components(graph) if n_nodes > 0 else 0,
        "n_intervals":       n_intervals,
        "overlap":           overlap,
        "dbscan_eps":        round(float(dbscan_eps), 4),
        "max_node_size":     int(max(all_sizes)) if all_sizes else 0,
        "avg_node_size":     round(float(np.mean(all_sizes)), 1) if all_sizes else 0.0,
        "l_min":             round(l_min, 4),
        "l_max":             round(l_max, 4),
        "anomaly_threshold": cfg.anomaly_high,
        "anomaly_base_rate": round(base_rate, 4),
        "risk_high_lift":    cfg.high_lift,
        "risk_watch_lift":   cfg.watch_lift,
        **mapper_risk_counts(nodes_out, cfg),
    }

    # Lens histogram clipped to the 2nd-98th percentile so extreme outliers
    # don't squash the chart; never cropped below the actual Mapper range.
    p2, p98 = float(np.percentile(lens, 2)), float(np.percentile(lens, 98))
    lens_clip = lens[(lens >= p2) & (lens <= p98)]
    n_bins = min(30, max(10, n_intervals * 2))
    use_clip = len(lens_clip) > 10
    hist_counts, hist_edges = np.histogram(lens_clip if use_clip else lens, bins=n_bins,
                                           range=(p2, p98) if use_clip else None)
    lens_histogram = [{"start": round(float(hist_edges[i]), 4), "end": round(float(hist_edges[i + 1]), 4),
                       "count": int(hist_counts[i])} for i in range(len(hist_counts))]
    interval_bands = [{"lo": round(float(lo), 4), "hi": round(float(lo + width), 4)} for lo in interval_starts]

    log.info("Mapper: %d nodes, %d edges, %d components", n_nodes, stats["n_edges"], stats["n_components"])
    return {"nodes": nodes_out, "edges": edges_out, "stats": stats,
            "lens_histogram": lens_histogram, "interval_bands": interval_bands}

