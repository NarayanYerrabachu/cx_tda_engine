"""The full pipeline: one call from a DataFrame to every finding list and the Mapper graph.

Two kinds of work are separated on purpose:

* **lens-independent** (normalise, persistent homology, clustering, anomaly
  scores, suspicious, relationships, drift, topology) is computed once per
  (feature set, row count) and cached in a :class:`BaseCache`;
* **lens-dependent** (lens transform + Mapper graph) is recomputed on every call.

So switching the lens on a 70k-row dataset is a sub-second Mapper rebuild
instead of a 40-second re-run.

A long run can be abandoned between stages through ``should_abort``: once it
returns True the run raises :class:`PipelineSuperseded` and leaves nothing in
the cache, so a newer dataset's run never inherits stale arrays.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .anomalies import detect_anomalies
from .clustering import compute_clusters
from .config import TDAConfig, resolve
from .drift import compute_drift
from .findings import build_suspicious_findings
from .homology import compute_persistent_homology
from .lenses import get_lens
from .mapper import build_mapper_graph
from .preprocessing import normalized_features
from .relationships import compute_relationships
from .types import PipelineResult, SuspiciousRule

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """What the engine needs to know about a table beyond its numeric features.

    Everything is optional: with the defaults the pipeline is purely numeric
    and titles records by their row position.
    """

    #: Column holding the record identifier (string-cast); None = row position.
    id_col: str | None = None
    #: Up to two categorical columns used to title clusters and anomalies and
    #: to build the relationship graph (``a`` x ``b`` co-occurrence).
    group_cols: tuple[str, ...] = ()
    #: Numeric column (e.g. a year) that splits early vs. late for drift.
    time_col: str | None = None
    #: Columns copied into ``extra`` of anomaly and suspicious findings for display.
    display_cols: tuple[str, ...] = ()
    #: Columns summarised (mean/std) per cluster theme.
    stat_cols: tuple[str, ...] = ()
    #: Raw metric columns for drift, ``{column: label}``; needs ``time_col``.
    metric_cols: dict[str, str] = field(default_factory=dict)
    #: Rules deciding which records are suspicious (default: score >= anomaly_high).
    suspicious_rules: tuple[SuspiciousRule, ...] | None = None
    #: Rows the relationship co-occurrence is counted over (default: all rows).
    relationship_basis: Callable[[pd.DataFrame], np.ndarray] | None = None
    relationship_type: str = "cooccurrence"

    def record_ids(self, df: pd.DataFrame) -> list[str]:
        if self.id_col and self.id_col in df.columns:
            return df[self.id_col].astype(str).tolist()
        return [str(i) for i in range(len(df))]


class PipelineSuperseded(Exception):
    """The run was abandoned between stages because ``should_abort()`` turned true."""


def _check_abort(should_abort: Callable[[], bool] | None) -> None:
    if should_abort is not None and should_abort():
        raise PipelineSuperseded()


class BaseCache:
    """Lens-independent results keyed by (sorted feature columns, row count).

    The key deliberately ignores cell values: hashing a large frame on every
    call would cost more than it saves. Call :meth:`clear` when the dataset
    changes; :meth:`seed` restores persisted results after a restart.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], dict[str, Any]] = {}

    @staticmethod
    def key(df: pd.DataFrame, feature_cols: list[str]) -> tuple[Any, ...]:
        return (tuple(sorted(c for c in feature_cols if c in df.columns)), len(df))

    def get(self, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any] | None:
        return self._store.get(self.key(df, feature_cols))

    def seed(self, df: pd.DataFrame, feature_cols: list[str], base: dict[str, Any]) -> None:
        self._store[self.key(df, feature_cols)] = base

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


#: Process-wide default cache used when ``run_full_pipeline`` gets no ``cache``.
DEFAULT_CACHE = BaseCache()


def clear_base_cache() -> None:
    """Drop the default cache. Call when the underlying dataset changes."""
    DEFAULT_CACHE.clear()


def cached_base(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any] | None:
    """The default cache's lens-independent results for this dataset, if computed."""
    return DEFAULT_CACHE.get(df, feature_cols)


def seed_base_cache(df: pd.DataFrame, feature_cols: list[str], base: dict[str, Any]) -> None:
    """Put restored lens-independent results into the default cache."""
    DEFAULT_CACHE.seed(df, feature_cols, base)


def compute_base(
    df: pd.DataFrame,
    feature_cols: list[str],
    spec: DatasetSpec | None = None,
    config: TDAConfig | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Every lens-independent stage, in order; ``should_abort`` is polled between them."""
    cfg = resolve(config)
    spec = spec or DatasetSpec()
    X_norm = normalized_features(df, feature_cols)
    record_ids = spec.record_ids(df)
    group_cols = [c for c in spec.group_cols if c in df.columns]
    display_cols = list(spec.display_cols)

    tda_info = compute_persistent_homology(X_norm, config=cfg)
    log.info("compute_base: homology done")
    _check_abort(should_abort)

    cluster_labels, themes = compute_clusters(X_norm, df, record_ids, group_cols, list(spec.stat_cols),
                                              config=cfg)
    log.info("compute_base: clusters done")
    _check_abort(should_abort)

    anomalies, combined, iso_scores, topo_scores = detect_anomalies(
        X_norm, df, record_ids, cluster_labels, feature_cols, group_cols, display_cols, config=cfg)
    n_anomalies_total = int((np.asarray(combined) >= cfg.anomaly_high).sum())
    log.info("compute_base: anomalies done (%d shown of %d)", len(anomalies), n_anomalies_total)
    _check_abort(should_abort)

    suspicious_all = build_suspicious_findings(
        df, combined, record_ids, top_k=None, iso_scores=iso_scores, topo_scores=topo_scores,
        rules=list(spec.suspicious_rules) if spec.suspicious_rules is not None else None,
        group_cols=group_cols, display_cols=display_cols, config=cfg)
    suspicious = suspicious_all[:cfg.suspicious_top_k]
    log.info("compute_base: suspicious done (%d shown of %d)", len(suspicious), len(suspicious_all))
    _check_abort(should_abort)

    basis = spec.relationship_basis(df) if spec.relationship_basis is not None else None
    relationships, graph_nodes, graph_edges = compute_relationships(
        df, record_ids, cluster_labels, group_cols, combined_scores=combined,
        suspicious_ids={str(f["sources"][0]) for f in suspicious_all if f.get("sources")},
        basis_mask=basis, relationship_type=spec.relationship_type, config=cfg)
    log.info("compute_base: relationships done (%d)", len(relationships))
    _check_abort(should_abort)

    drift = compute_drift(df, X_norm, cluster_labels, feature_cols, spec.time_col, spec.metric_cols or None)
    log.info("compute_base: drift done")

    topology = [{
        "kind": "topology", "title": lp["label"], "score": round(float(lp["persistence"]), 4),
        "sources": lp.get("sources", []),
        "detail": (f"H₁ loop: birth={lp['birth']:.4f}, death={lp['death']:.4f}, "
                   f"persistence={lp['persistence']:.4f}."),
        "extra": lp,
    } for lp in tda_info.get("h1_features", [])]

    return {
        "X_norm":             X_norm,
        "record_ids":         record_ids,
        "cluster_labels":     cluster_labels,
        "combined_scores":    combined,
        "iso_scores":         iso_scores,
        "topo_scores":        topo_scores,
        "feat_cols":          feature_cols,
        "cat_cols":           group_cols,
        "themes":             themes,
        "anomalies":          anomalies,
        "suspicious":         suspicious,
        "n_anomalies_total":  n_anomalies_total,
        "n_suspicious_total": len(suspicious_all),
        "relationships":      relationships,
        "drift":              drift,
        "topology":           topology,
        "graph_nodes":        graph_nodes,
        "graph_edges":        graph_edges,
        "tda_info":           tda_info,
        "n_docs":             len(df),
        "n_clusters":         len(set(cluster_labels.tolist())) - (1 if -1 in cluster_labels else 0),
        "n_noise":            int((cluster_labels == -1).sum()),
    }


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
    base = hit if hit is not None else compute_base(df, feat_cols, spec, cfg, should_abort)
    _check_abort(should_abort)   # before caching: a superseded run must not leave its arrays behind
    if store is not None and hit is None:
        store.seed(df, feat_cols, base)

    X_norm, record_ids, cluster_labels = base["X_norm"], base["record_ids"], base["cluster_labels"]

    lens = get_lens(lens_name, **(lens_kwargs or {}))
    lens_vals = np.asarray(lens.fit_transform(X_norm), dtype=float)
    if lens_vals.ndim > 1:
        lens_vals = lens_vals[:, 0]

    eff_n_intervals = max(n_intervals, cfg.mapper_min_n_intervals)
    eff_overlap = max(overlap, cfg.mapper_min_overlap)
    if eff_n_intervals != n_intervals or eff_overlap != overlap:
        log.info("Mapper resolution floors applied: n_intervals %d->%d, overlap %.2f->%.2f",
                 n_intervals, eff_n_intervals, overlap, eff_overlap)

    mapper = build_mapper_graph(X_norm, lens_vals, eff_n_intervals, eff_overlap, record_ids,
                                anomaly_scores=base["combined_scores"], config=cfg)
    elapsed = round(time.time() - t0, 2)
    log.info("run_full_pipeline: lens=%s cached_base=%s %.2fs (%d mapper nodes)",
             lens_name, hit is not None, elapsed, len(mapper["nodes"]))

    meta = {
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
        "cached_base":         hit is not None,
        "elapsed_s":           elapsed,
        "mapper_n_intervals":  eff_n_intervals,
        "mapper_overlap":      eff_overlap,
        "mapper_requested":    {"n_intervals": n_intervals, "overlap": overlap},
        "mapper_min_n_intervals": cfg.mapper_min_n_intervals,
        "mapper_min_overlap":  cfg.mapper_min_overlap,
        "seed":                cfg.seed,
        "contamination":       cfg.effective_contamination,
        "engine_version":      _version(),
    }
    return {
        "meta":            meta,
        "themes":          base["themes"],
        "anomalies":       base["anomalies"],
        "suspicious":      base["suspicious"],
        "relationships":   base["relationships"],
        "drift":           base["drift"],
        "topology":        base["topology"],
        "graph": {"nodes": base["graph_nodes"], "edges": base["graph_edges"],
                  "mapper_nodes": mapper["nodes"], "mapper_edges": mapper["edges"]},
        "documents":       [],
        "record_ids":      record_ids,
        "lens_values":     lens_vals.tolist(),
        "cluster_labels":  cluster_labels.tolist(),
        "combined_scores": base["combined_scores"].tolist() if len(base["combined_scores"]) else [],
    }


def _version() -> str:
    from . import __version__

    return __version__
