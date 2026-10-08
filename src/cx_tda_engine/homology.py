"""Persistent homology (H0 / H1) with ripser on a PCA projection."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from .config import TDAConfig, resolve
from .risk import loop_class

log = logging.getLogger(__name__)


def homology_projection(X_norm: np.ndarray, config: TDAConfig | None = None) -> np.ndarray:
    """PCA projection (<= 5 components, seeded) that persistent homology runs on.

    Exposed so a significance test (e.g. Monte Carlo) can test exactly the
    point cloud behind the persistence diagram.
    """
    cfg = resolve(config)
    n_comp = max(1, min(5, X_norm.shape[1], len(X_norm)))
    return PCA(n_components=n_comp, random_state=cfg.seed).fit_transform(X_norm)


def compute_persistent_homology(X_norm: np.ndarray, sample_n: int | None = None,
                                config: TDAConfig | None = None) -> dict[str, Any]:
    """H0/H1 persistent homology of (a seeded sample of) ``X_norm``.

    Betti numbers reported here are NOT the raw bar counts (the raw H0 count
    equals the sample size by construction: every point is its own component
    at epsilon = 0). Instead they are counted at a filtration scale
    ``noise_threshold`` derived from the H1 persistence distribution, so
    ``betti_0`` counts long-lived connected components and ``betti_1`` counts
    loops that survive the noise band. Raw counts are still exposed as
    ``betti_0_raw`` / ``betti_1_raw`` for the persistence-diagram view.

    Never raises: when ripser is missing or fails, ``available`` is False and
    the remaining fields hold neutral defaults.
    """
    cfg = resolve(config)
    sample_n = cfg.sample_n if sample_n is None else sample_n
    result: dict[str, Any] = {
        "available":         False,
        "betti_0":           1,
        "betti_1":           0,
        "betti_0_raw":       0,
        "betti_1_raw":       0,
        "noise_threshold":   0.0,
        "max_persistence":   0.0,
        "loop_class":        None,
        "persistence_ratio": None,
        "h0_features":       [],
        "h1_features":       [],
    }
    try:
        from ripser import ripser as _ripser

        n = len(X_norm)
        X_pca = homology_projection(X_norm, cfg)
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(n, min(sample_n, n), replace=False)
        X_s = X_pca[idx]

        dgms = _ripser(X_s, maxdim=1)["dgms"]

        # ── H0: connected components ─────────────────────────────────────────
        h0        = dgms[0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        h0_inf    = h0[~np.isfinite(h0[:, 1])]

        h0_feats = [
            {"dim": 0, "birth": float(r[0]), "death": float(r[1]),
             "persistence": float(r[1] - r[0]), "infinite": False}
            for r in h0_finite[:50]
        ] + [
            {"dim": 0, "birth": float(r[0]), "death": None, "persistence": None, "infinite": True}
            for r in h0_inf[:1]              # only the one true infinite component
        ]

        # ── H1: loops ────────────────────────────────────────────────────────
        h1     = dgms[1]
        finite = h1[np.isfinite(h1[:, 1])]
        betti1_raw = len(finite)
        pers   = finite[:, 1] - finite[:, 0] if betti1_raw > 0 else np.array([])
        max_p  = float(pers.max()) if betti1_raw > 0 else 0.0

        # Noise band: data-adaptive threshold on H1 persistence. Everything
        # below is treated as sampling noise and neither counted nor listed as
        # a loop finding. Rule: 2x median persistence (a "top of the noise
        # band" heuristic robust to skew) with a small absolute floor so an
        # over-uniform diagram does not degenerate to zero.
        if betti1_raw > 0:
            median_p = float(np.median(pers))
            noise_threshold = max(2.0 * median_p, 1e-3)
        else:
            median_p = 0.0
            noise_threshold = 0.0

        significant_mask = pers > noise_threshold if betti1_raw > 0 else np.array([], dtype=bool)
        betti1_sig = int(significant_mask.sum())

        # Keep top-200 bars for the diagram; the top-10 significant ones get labels.
        top200 = finite[np.argsort(-pers)[:200]] if betti1_raw > 0 else []
        h1_feats = [
            {
                "dim":         1,
                "birth":       float(r[0]),
                "death":       float(r[1]),
                "persistence": float(r[1] - r[0]),
                "infinite":    False,
                "label": (f"Loop {i + 1} (pers {r[1] - r[0]:.4f})"
                          if (r[1] - r[0]) > noise_threshold and i < 10 else ""),
                "sources":     [],
                "significant": bool((r[1] - r[0]) > noise_threshold),
            }
            for i, r in enumerate(top200)
        ]

        # Axis bound from H1 deaths, not H0: H0 deaths are raw PCA distances,
        # orders of magnitude larger, and would squash H1 into the corner.
        h1_death_max = float(finite[:, 1].max()) if betti1_raw > 0 else max_p
        diagram_max  = round(h1_death_max * 1.08, 4)

        # beta_0 at the noise scale: components that have NOT merged by
        # epsilon = noise_threshold (a meaningful count, unlike the raw total).
        if noise_threshold > 0:
            n_alive_finite = int((h0_finite[:, 1] > noise_threshold).sum())
        else:
            n_alive_finite = len(h0_finite)
        betti0_at_scale = n_alive_finite + len(h0_inf)
        betti0_raw = len(h0_finite) + len(h0_inf)

        result.update({
            "available":             True,
            "betti_0":               betti0_at_scale,
            "betti_1":               betti1_sig,
            "betti_0_raw":           betti0_raw,
            "betti_1_raw":           betti1_raw,
            "noise_threshold":       round(noise_threshold, 6),
            "median_h1_persistence": round(median_p, 6),
            "max_persistence":       round(max_p, 4),
            "h0_features":           h0_feats,
            "h1_features":           [f for f in h1_feats if f["label"]],
            "diagram_h0":            h0_feats,
            "diagram_h1":            h1_feats,
            "diagram_max":           diagram_max,
            "loop_class":            loop_class(betti1_sig, cfg),
            "persistence_ratio":     (round(max_p / noise_threshold, 4) if noise_threshold > 0 else None),
        })
        log.info("homology: b0=%d (raw %d) b1=%d (raw %d) noise_threshold=%.4f max_pers=%.4f",
                 betti0_at_scale, betti0_raw, betti1_sig, betti1_raw, noise_threshold,
                 result["max_persistence"])
    except Exception as exc:  # ripser missing or numerical failure: degrade, never crash
        log.warning("Persistent homology unavailable: %s", exc)
    return result
