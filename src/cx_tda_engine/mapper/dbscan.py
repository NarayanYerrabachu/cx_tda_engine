"""Memory-safe DBSCAN for one Mapper interval, and the eps heuristic for the exploration graph.

DBSCAN's radius-neighbour graph grows with the square of the points in a
dense neighbourhood; on a 68k-row dataset one interval reached ~9 GB and the
process was OOM-killed. ``dbscan_capped`` estimates the pair count first and,
only when it exceeds the configured budget, fits on a subsample and assigns
the rest to the nearest core sample. Node membership stays complete.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from ..config import TDAConfig, resolve

log = logging.getLogger(__name__)

_MIN_FIT = 2000
AUTO_DEFAULT_EPS = 0.7


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

