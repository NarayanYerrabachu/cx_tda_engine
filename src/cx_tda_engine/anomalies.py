"""Per-record anomaly scores: Isolation Forest blended with a topological score."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import TDAConfig, resolve
from .risk import priority

_MISSING = ("nan", "None", "Unknown", "N/A", "")


def _clean_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return round(float(v), 4)
    return str(v)


def anomaly_scores(X_norm: np.ndarray, cluster_labels: np.ndarray,
                   config: TDAConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(combined, iso, topo)`` scores for every row, combined scaled to [0, 1].

    * iso: Isolation Forest anomaly score (higher = more anomalous).
    * topo: distance from the row's cluster centroid, scaled per cluster to
      [0, 1]; DBSCAN noise points get 1.0.
    * combined: ``iso_weight * iso + (1 - iso_weight) * topo``, min-max scaled.
    """
    cfg = resolve(config)
    n = len(X_norm)
    iso = IsolationForest(contamination=cfg.effective_contamination, random_state=cfg.seed, n_jobs=cfg.n_jobs)
    iso.fit(X_norm)
    iso_scores = -iso.score_samples(X_norm)

    topo_scores = np.zeros(n)
    for cid in set(cluster_labels.tolist()):
        if cid == -1:
            continue
        mask = cluster_labels == cid
        centroid = X_norm[mask].mean(axis=0)
        dists = np.linalg.norm(X_norm[mask] - centroid, axis=1)
        topo_scores[mask] = dists / (dists.max() + 1e-9)
    topo_scores[cluster_labels == -1] = 1.0

    combined = cfg.iso_weight * iso_scores + (1.0 - cfg.iso_weight) * topo_scores
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)
    return combined, iso_scores, topo_scores


def detect_anomalies(
    X_norm: np.ndarray,
    df: pd.DataFrame,
    record_ids: list[str],
    cluster_labels: np.ndarray,
    feature_cols: list[str],
    group_cols: list[str] | None = None,
    display_cols: list[str] | None = None,
    top_k: int | None = None,
    config: TDAConfig | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Score every row and describe the ``top_k`` highest as ``kind="anomaly"`` findings.

    Returns ``(findings, combined, iso, topo)``; the three arrays cover ALL rows.
    A finding's title is the record id plus the values of the first two
    ``group_cols``; ``extra`` carries the scores, the three features that
    deviate most (``flagged_by``), the ``display_cols`` and the first six
    feature values, and a ``priority`` label.
    """
    cfg = resolve(config)
    top_k = cfg.anomaly_top_k if top_k is None else top_k
    group_cols = [c for c in (group_cols or []) if c in df.columns]
    display_cols = [c for c in (display_cols or []) if c in df.columns]

    combined, iso_scores, topo_scores = anomaly_scores(X_norm, cluster_labels, cfg)
    top_idx = np.argsort(combined)[::-1][:top_k]

    feat_means = X_norm.mean(axis=0)
    feat_stds  = X_norm.std(axis=0) + 1e-9

    findings = []
    for i in top_idx:
        pid = record_ids[i]
        row = df.iloc[i]
        z_scores = np.abs(X_norm[i] - feat_means) / feat_stds
        flagged  = [feature_cols[j] for j in np.argsort(-z_scores)[:3] if j < len(feature_cols)]

        parts = []
        for c in group_cols[:2]:
            v = row.get(c, "")
            if v is not None and str(v) not in _MISSING:
                parts.append(str(v))
        title = pid + (" — " + ", ".join(parts) if parts else "")

        extra: dict[str, Any] = {}
        for c in display_cols + feature_cols[:6]:
            if c in df.columns:
                v = _clean_value(row.get(c))
                if v is not None:
                    extra[c] = v
        extra["priority"] = priority(float(combined[i]), cfg)

        findings.append({
            "kind":    "anomaly",
            "title":   title,
            "score":   round(float(combined[i]), 4),
            "sources": [pid],
            "detail":  (f"Anomaly score: {combined[i]:.4f}. Isolation Forest: {iso_scores[i]:.4f}. "
                        f"Topo score: {topo_scores[i]:.4f}. Cluster: {int(cluster_labels[i])}."),
            "extra": {
                "iso_score":  round(float(iso_scores[i]), 4),
                "ae_score":   None,
                "topo_score": round(float(topo_scores[i]), 4),
                "flagged_by": flagged,
                **extra,
            },
        })
    return findings, combined, iso_scores, topo_scores
