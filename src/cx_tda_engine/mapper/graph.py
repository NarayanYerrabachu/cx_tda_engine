"""The pipeline Mapper graph: the one :func:`run_full_pipeline` returns.

Cover: ``n_intervals`` equal slices of the lens range, each widened by
``overlap`` on both sides. Local clustering: fixed-eps DBSCAN on a 2-D PCA of
the interval, so the result is stable across datasets and cheap enough to
rebuild on every lens switch. Nodes carry anomaly statistics and a ``state``
computed over ALL their members, never over the capped ``sources`` sample.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from ..config import TDAConfig, resolve
from ..core.risk import graph_node_state

log = logging.getLogger(__name__)


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

