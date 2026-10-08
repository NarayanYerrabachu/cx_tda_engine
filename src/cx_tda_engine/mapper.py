"""Mapper (Singh, Memoli, Carlsson 2007) in two flavours.

Both follow the same algorithm: slice the lens image into overlapping
intervals, cluster the preimage of every interval, make each cluster a node
and connect nodes that share records.

* :func:`build_mapper_graph` is the pipeline's graph: a fixed-eps DBSCAN on a
  2-D PCA of each interval, nodes carry ``state`` and ``node_score`` for the
  topology view. Cheap and stable, used by :func:`cx_tda_engine.run_full_pipeline`.
* :func:`run_mapper` is the exploration graph: interval-wise standardisation,
  auto-tuned eps, risk tiers, neighbour lists, lens histogram and interval
  bands for a visualisation panel.

Both guard against DBSCAN's quadratic neighbour graph on dense intervals by
fitting on a subsample and assigning the remaining members afterwards; node
membership is always complete, only the fit is subsampled.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import TDAConfig, resolve
from .risk import graph_node_state, mapper_node_risk_level, mapper_risk_counts, node_risk_level

log = logging.getLogger(__name__)


# ── Pipeline Mapper ───────────────────────────────────────────────────────────

def build_mapper_graph(
    X_norm: np.ndarray,
    lens_values: np.ndarray,
    n_intervals: int = 10,
    overlap: float = 0.5,
    record_ids: list[str] | None = None,
    anomaly_scores: np.ndarray | None = None,
    config: TDAConfig | None = None,
) -> dict[str, Any]:
    """Mapper graph for the pipeline: ``{"nodes": [...], "edges": [...]}``.

    When ``anomaly_scores`` (per-record combined scores aligned with the rows
    of ``X_norm``) is given, every node also carries ``avg_anomaly``,
    ``max_anomaly``, ``frac_anomalous`` (share of members >= ``anomaly_high``),
    ``n_anomalous``, ``node_score``, ``lift`` and ``state`` computed over ALL
    members, never over the capped ``sources`` sample.
    """
    cfg = resolve(config)
    scores = None
    if anomaly_scores is not None and len(anomaly_scores) == len(X_norm):
        scores = np.asarray(anomaly_scores, dtype=float)
    lens_values = np.asarray(lens_values, dtype=float).ravel()
    lo, hi   = lens_values.min(), lens_values.max()
    span     = hi - lo + 1e-9
    step     = span / n_intervals
    half_ov  = step * overlap / 2

    interval_members: list[list[int]] = []
    for k in range(n_intervals):
        c    = lo + step * (k + 0.5)
        lo_k = c - step / 2 - half_ov
        hi_k = c + step / 2 + half_ov
        members = np.where((lens_values >= lo_k) & (lens_values <= hi_k))[0].tolist()
        if members:
            interval_members.append(members)

    mapper_nodes: list[dict[str, Any]] = []
    for im_idx, members in enumerate(interval_members):
        X_sub = X_norm[members]
        if len(X_sub) < 2:
            mapper_nodes.append({"members": members, "interval": im_idx, "cluster": 0})
            continue
        n_comp_sub = max(1, min(2, X_sub.shape[1], X_sub.shape[0]))
        pca_sub = PCA(n_components=n_comp_sub, random_state=cfg.seed).fit_transform(X_sub)

        if len(pca_sub) > cfg.mapper_fit_cap:
            # Fit on a seeded subsample, assign everyone else to the nearest centroid.
            rng_i = np.random.default_rng(cfg.seed + im_idx)
            fit_idx = rng_i.choice(len(pca_sub), cfg.mapper_fit_cap, replace=False)
            fit_lbs = DBSCAN(eps=1.2, min_samples=1).fit(pca_sub[fit_idx]).labels_
            cids = np.array([c for c in sorted(set(fit_lbs)) if c != -1])
            if len(cids) == 0:
                lbs = np.zeros(len(pca_sub), dtype=int)
            else:
                cents = np.vstack([pca_sub[fit_idx][fit_lbs == c].mean(axis=0) for c in cids])
                d = np.linalg.norm(pca_sub[:, None, :] - cents[None, :, :], axis=2)
                lbs = cids[np.argmin(d, axis=1)]
        else:
            lbs = DBSCAN(eps=1.2, min_samples=1).fit(pca_sub).labels_

        for cid in sorted(set(lbs)):
            mask  = lbs == cid
            group = [members[j] for j in range(len(members)) if mask[j]]
            if group:
                mapper_nodes.append({"members": group, "interval": im_idx, "cluster": int(cid)})

    # Only nodes from the same or adjacent intervals can share members.
    mapper_edges: list[dict[str, Any]] = []
    member_sets = [set(nd["members"]) for nd in mapper_nodes]
    for i in range(len(mapper_nodes)):
        for j in range(i + 1, len(mapper_nodes)):
            if abs(mapper_nodes[i]["interval"] - mapper_nodes[j]["interval"]) > 1:
                continue
            shared = member_sets[i] & member_sets[j]
            if shared:
                mapper_edges.append({"source": i, "target": j, "weight": len(shared)})

    rids = record_ids or [str(k) for k in range(len(X_norm))]
    base_rate = float((scores >= cfg.anomaly_high).mean()) if scores is not None and len(scores) else 0.0
    graph_nodes = []
    for i, nd in enumerate(mapper_nodes):
        members  = nd["members"]
        gn: dict[str, Any] = {
            "id":        f"node-{i}",
            "size":      len(members),
            "sources":   [rids[m] for m in members][:20],
            "interval":  nd["interval"],
            "cluster":   nd["cluster"],
            "lens_mean": float(lens_values[members].mean()) if members else 0.0,
        }
        if scores is not None and members:
            ms   = scores[members]
            avg  = float(ms.mean())
            mx   = float(ms.max())
            frac = float((ms >= cfg.anomaly_high).mean())
            w    = cfg.score_frac_weight
            gn.update({
                "avg_anomaly":    round(avg, 4),
                "max_anomaly":    round(mx, 4),
                "frac_anomalous": round(frac, 4),
                "n_anomalous":    int((ms >= cfg.anomaly_high).sum()),
                "node_score":     round(min(w * frac + (1.0 - w) * avg, 1.0), 4),
                "lift":           round(frac / base_rate, 3) if base_rate else None,
                "state":          graph_node_state(frac, base_rate, cfg),
            })
        graph_nodes.append(gn)

    return {"nodes": graph_nodes, "edges": mapper_edges}


# ── Exploration Mapper ────────────────────────────────────────────────────────

_MIN_FIT = 2000
_AUTO_DEFAULT_EPS = 0.7


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


def _estimated_pairs(X: np.ndarray, eps: float, seed: int) -> float:
    """Estimate how many (i, j) neighbour pairs within eps DBSCAN would store."""
    n = len(X)
    if n <= 3000:
        return 0.0
    rng = np.random.default_rng(seed)
    probe = X[rng.choice(n, 1000, replace=False)]
    nn = NearestNeighbors(radius=eps).fit(X)
    counts = np.array([len(ix) for ix in nn.radius_neighbors(probe, return_distance=False)])
    return float(counts.mean()) * n


def dbscan_capped(X: np.ndarray, eps: float, min_samples: int, seed: int,
                  config: TDAConfig | None = None) -> np.ndarray:
    """DBSCAN labels for X: all rows, unless memory would blow up.

    Before clustering we estimate the neighbour-pair count DBSCAN would hold.
    Only when it exceeds ``config.mapper_max_pairs`` (or ``run_mapper_fit_cap``
    forces a fixed cap) is DBSCAN fitted on a subsample sized to stay under the
    limit; every other row is assigned to the cluster of its nearest core
    sample, or to noise if none is within eps.
    """
    cfg = resolve(config)
    n = len(X)
    fit_n = n
    if cfg.run_mapper_fit_cap > 0 and n > cfg.run_mapper_fit_cap:
        fit_n = cfg.run_mapper_fit_cap
    else:
        pairs = _estimated_pairs(X, eps, seed)
        if pairs > cfg.mapper_max_pairs:
            fit_n = max(_MIN_FIT, int(n * (cfg.mapper_max_pairs / pairs) ** 0.5))
            log.warning("Mapper interval: %d rows, ~%.0fM neighbour pairs > limit %.0fM: fitting DBSCAN on "
                        "%d rows, assigning the rest to the nearest core",
                        n, pairs / 1e6, cfg.mapper_max_pairs / 1e6, fit_n)
    if fit_n >= n:
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(n, fit_n, replace=False)
    # Scale min_samples to the subsample so density thresholds stay comparable.
    ms = max(2, int(round(min_samples * fit_n / n))) if min_samples > 2 else min_samples
    db = DBSCAN(eps=eps, min_samples=ms).fit(X[fit_idx])
    core = db.core_sample_indices_
    labels = np.full(n, -1, dtype=int)
    if len(core) == 0:
        return labels
    core_X = X[fit_idx][core]
    core_lb = db.labels_[core]
    dist, nn = NearestNeighbors(n_neighbors=1).fit(core_X).kneighbors(X)
    within = dist[:, 0] <= eps
    labels[within] = core_lb[nn[within, 0]]
    return labels


def auto_mapper_eps(X_norm: np.ndarray, min_samples: int, seed: int = 0) -> float:
    """eps for the interval DBSCANs from k-NN distances of a 2,000-row sample."""
    n = len(X_norm)
    sample_n = min(n, 2000)
    sidx = np.random.default_rng(seed).choice(n, sample_n, replace=False)
    Xs = X_norm[sidx]
    nn = NearestNeighbors(n_neighbors=min(min_samples + 1, len(Xs))).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    knn = np.sort(dists[:, -1])
    lo  = float(np.percentile(knn, 65))
    hi  = float(np.percentile(knn, 85))
    eps = float(np.sqrt(lo * hi)) if lo > 0 else hi
    log.info("Mapper auto_eps: k=%d p65=%.4f p85=%.4f -> eps=%.4f", min_samples, lo, hi, eps)
    return eps


def run_mapper(
    X_norm: np.ndarray,
    lens_values: np.ndarray,
    n_intervals: int = 10,
    overlap: float = 0.4,
    dbscan_eps: float = _AUTO_DEFAULT_EPS,
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

    if abs(dbscan_eps - _AUTO_DEFAULT_EPS) < 1e-6:
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
