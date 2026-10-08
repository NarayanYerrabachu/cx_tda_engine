"""Global DBSCAN clustering on a PCA projection, with data-adaptive eps."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ..config import TDAConfig, resolve

log = logging.getLogger(__name__)

_EPS_EXACT_MAX = 200_000
_MISSING = ("nan", "None", "Unknown", "N/A", "")


def auto_eps(X: np.ndarray, k: int = 5, lo_pct: float = 70, hi_pct: float = 90, seed: int = 42) -> float:
    """Estimate a DBSCAN eps from k-th nearest-neighbour distances.

    Takes the geometric mean of the ``lo_pct`` and ``hi_pct`` percentiles so
    the result adapts to both sparse and dense datasets.

    k-NN distances are measured on the FULL point set. Measuring them on a
    subsample of size m inflates eps by roughly (n/m)^(1/d), because a
    subsample is sparser; on a 68k-row, 5-feature dataset that was a 1.86x
    overestimate which collapsed 97 % of rows into one cluster. A KD-tree
    query for small k is cheap, so we only subsample past 200k rows and then
    apply the density correction explicitly.

    Raises ``ValueError`` when every record has identical features.
    """
    n, d = len(X), (X.shape[1] if X.ndim > 1 else 1)
    if n <= _EPS_EXACT_MAX:
        Xs, correction = X, 1.0
    else:
        m = _EPS_EXACT_MAX
        Xs = X[np.random.default_rng(seed).choice(n, m, replace=False)]
        correction = (m / n) ** (1.0 / max(d, 1))

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xs))).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    knn = np.sort(dists[:, -1])
    lo = float(np.percentile(knn, lo_pct))
    hi = float(np.percentile(knn, hi_pct))
    eps = float(np.sqrt(lo * hi)) if lo > 0 else hi
    if eps <= 0:
        # Most points coincide (e.g. only a few categorical columns): use the
        # typical distance between the points that do differ.
        positive = dists[:, 1:][dists[:, 1:] > 0]
        if positive.size == 0:
            raise ValueError(
                "Every record has the same feature values, so there is no structure to analyse. "
                "Provide numeric columns that vary between records.")
        eps = float(np.median(positive))
    eps *= correction
    log.info("auto_eps: n=%d k=%d p%d=%.4f p%d=%.4f correction=%.3f -> eps=%.4f",
             n, k, lo_pct, lo, hi_pct, hi, correction, eps)
    return eps


def compute_clusters(
    X_norm: np.ndarray,
    df: pd.DataFrame,
    record_ids: list[str],
    group_cols: list[str] | None = None,
    stat_cols: list[str] | None = None,
    eps: float | None = None,
    min_samples: int | None = None,
    config: TDAConfig | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """DBSCAN on a <= 5-D PCA of ``X_norm``.

    Returns ``(labels, themes)``: one label per row (-1 = noise) and one
    ``kind="theme"`` finding per cluster. A theme's title names the most common
    value of each of the first two ``group_cols``; ``stat_cols`` get a
    mean/std summary in ``extra.stats``.

    ``eps`` left at None (or at ``config.dbscan_eps``) is auto-tuned: a fixed
    value calibrated for one dataset fails on others.
    """
    cfg = resolve(config)
    group_cols = [c for c in (group_cols or []) if c in df.columns]
    stat_cols = [c for c in (stat_cols or []) if c in df.columns]
    min_samples = cfg.dbscan_min_samples if min_samples is None else min_samples

    n_samples, n_features = X_norm.shape
    n_comp = max(1, min(5, n_features, n_samples))
    X_pca = PCA(n_components=n_comp, random_state=cfg.seed).fit_transform(X_norm)

    if eps is None or eps == cfg.dbscan_eps:
        eps = auto_eps(X_pca, k=min_samples, seed=cfg.seed)

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=cfg.n_jobs).fit(X_pca).labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log.info("DBSCAN: %d clusters, %d noise", n_clusters, int((labels == -1).sum()))

    themes: list[dict[str, Any]] = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        rows = np.where(mask)[0]
        c_df = df.iloc[rows]

        label_parts = []
        for c in group_cols[:2]:
            counts = c_df[c].value_counts()          # drops NaN; may be empty
            if len(counts) and str(counts.index[0]) not in _MISSING:
                label_parts.append(str(counts.index[0]))

        stat_fields = {}
        for col in stat_cols:
            series = pd.to_numeric(c_df[col], errors="coerce")
            if series.notna().any():
                stat_fields[col] = {"mean": round(float(series.mean()), 3),
                                    "std": round(float(series.std()), 3) if len(series) > 1 else 0.0}

        themes.append({
            "kind":    "theme",
            "title":   f"Cluster {cid}" + (": " + " · ".join(label_parts[:3]) if label_parts else ""),
            "score":   0.0,
            "sources": [record_ids[j] for j in rows[:15]],
            "detail":  f"{int(mask.sum())} records in this cluster.",
            "extra":   {"cluster_id": int(cid), "n_records": int(mask.sum()), "stats": stat_fields},
        })

    return labels, themes
