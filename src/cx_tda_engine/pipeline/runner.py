"""``run_full_pipeline``: cached base + lens + Mapper, returned as the result contract."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import TDAConfig, resolve
from ..core.types import PipelineResult
from ..lenses import get_lens
from ..mapper import build_mapper_graph
from .cache import DEFAULT_CACHE, BaseCache
from .spec import DatasetSpec
from .stages import DEFAULT_STAGES, Stage, check_abort, compute_base

log = logging.getLogger(__name__)


def run_full_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    lens_name: str = "pca",
    n_intervals: int = 10,
    overlap: float = 0.5,
    *,
    spec: DatasetSpec | None = None,
    config: TDAConfig | None = None,
    cache: BaseCache | None = DEFAULT_CACHE,
    use_cache: bool = True,
    should_abort: Callable[[], bool] | None = None,
    lens_kwargs: dict[str, Any] | None = None,
    stages: tuple[Stage, ...] = DEFAULT_STAGES,
) -> PipelineResult:
    """Run everything and return the result contract (see ``docs/result_contract.md``).

    ``feature_cols`` not present in ``df`` are ignored; with none left, or
    fewer than ``config.min_rows`` rows, the result is ``{"error": ...}``
    instead of an exception, so a service can surface the message.

    ``n_intervals`` / ``overlap`` are floored at ``config.mapper_min_*``: too
    coarse a cover collapses to a single node. The effective values are in
    ``meta``.
    """
    cfg = resolve(config)
    spec = spec or DatasetSpec()
    t0 = time.time()

    feat_cols = [c for c in feature_cols if c in df.columns]
    if not feat_cols:
        return {"error": "No TDA feature columns available in DataFrame."}
    if len(df) < cfg.min_rows:
        return {"error": f"TDA analysis needs at least {cfg.min_rows} rows; got {len(df)}."}

    store = cache if (use_cache and cache is not None) else None
    hit = store.get(df, feat_cols) if store is not None else None
    base = hit if hit is not None else compute_base(df, feat_cols, spec, cfg, should_abort, stages)
    check_abort(should_abort)   # before caching: a superseded run must not leave its arrays behind
    if store is not None and hit is None:
        store.seed(df, feat_cols, base)

    lens_vals = np.asarray(get_lens(lens_name, **(lens_kwargs or {})).fit_transform(base["X_norm"]), dtype=float)
    if lens_vals.ndim > 1:
        lens_vals = lens_vals[:, 0]

    eff_n_intervals = max(n_intervals, cfg.mapper_min_n_intervals)
    eff_overlap = max(overlap, cfg.mapper_min_overlap)
    if eff_n_intervals != n_intervals or eff_overlap != overlap:
        log.info("Mapper resolution floors applied: n_intervals %d->%d, overlap %.2f->%.2f",
                 n_intervals, eff_n_intervals, overlap, eff_overlap)

    mapper = build_mapper_graph(base["X_norm"], lens_vals, eff_n_intervals, eff_overlap, base["record_ids"],
                                anomaly_scores=base["combined_scores"], config=cfg)
    elapsed = round(time.time() - t0, 2)
    log.info("run_full_pipeline: lens=%s cached_base=%s %.2fs (%d mapper nodes)",
             lens_name, hit is not None, elapsed, len(mapper["nodes"]))

    return {
        "meta":            _meta(base, cfg, lens_name, hit is not None, elapsed,
                                 n_intervals, overlap, eff_n_intervals, eff_overlap),
        "themes":          base["themes"],
        "anomalies":       base["anomalies"],
        "suspicious":      base["suspicious"],
        "relationships":   base["relationships"],
        "drift":           base["drift"],
        "topology":        base["topology"],
        "graph": {"nodes": base["graph_nodes"], "edges": base["graph_edges"],
                  "mapper_nodes": mapper["nodes"], "mapper_edges": mapper["edges"]},
        "documents":       [],
        "record_ids":      base["record_ids"],
        "lens_values":     lens_vals.tolist(),
        "cluster_labels":  base["cluster_labels"].tolist(),
        "combined_scores": base["combined_scores"].tolist() if len(base["combined_scores"]) else [],
    }


def _meta(base: dict[str, Any], cfg: TDAConfig, lens_name: str, cached: bool, elapsed: float,
          n_intervals: int, overlap: float, eff_n_intervals: int, eff_overlap: float) -> dict[str, Any]:
    from .. import __version__

    return {
        "n_docs":              base["n_docs"],
        "n_clusters":          base["n_clusters"],
        "n_noise_docs":        base["n_noise"],
        "n_anomalies":         len(base["anomalies"]),
        "n_suspicious":        len(base["suspicious"]),
        "n_anomalies_total":   base["n_anomalies_total"],
        "n_suspicious_total":  base["n_suspicious_total"],
        "anomaly_threshold":   cfg.anomaly_high,
        "n_relationships":     len(base["relationships"]),
        "n_drift":             len(base["drift"]),
        "n_topology":          len(base["topology"]),
        "graph_backend":       "in-memory",
        "ripser_available":    base["tda_info"]["available"],
        "theme_quality":       None,
        "n_themes":            base["n_clusters"],
        "tda":                 base["tda_info"],
        "numeric_features":    base["feat_cols"],
        "category_features":   base["cat_cols"],
        "data_type":           "tabular",
        "lens":                lens_name,
        "cached_base":         cached,
        "elapsed_s":           elapsed,
        "mapper_n_intervals":  eff_n_intervals,
        "mapper_overlap":      eff_overlap,
        "mapper_requested":    {"n_intervals": n_intervals, "overlap": overlap},
        "mapper_min_n_intervals": cfg.mapper_min_n_intervals,
        "mapper_min_overlap":  cfg.mapper_min_overlap,
        "seed":                cfg.seed,
        "contamination":       cfg.effective_contamination,
        "engine_version":      __version__,
    }
