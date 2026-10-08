"""The lens-independent work as a chain of stages over one shared context.

Each stage is a plain function ``(ctx) -> None`` that reads what earlier
stages produced and writes its own result onto the context. ``run_stages``
executes them in order and polls ``should_abort`` between them, so a run can
be abandoned cleanly when a newer dataset arrives. Adding a stage means
adding a function and a list entry, nothing else.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..analysis import (
    build_suspicious_findings,
    compute_clusters,
    compute_drift,
    compute_persistent_homology,
    compute_relationships,
    detect_anomalies,
)
from ..config import TDAConfig, resolve
from ..core.preprocessing import normalized_features
from ..core.types import make_finding
from .spec import DatasetSpec

log = logging.getLogger(__name__)


class PipelineSuperseded(Exception):
    """The run was abandoned between stages because ``should_abort()`` turned true."""


@dataclass
class BaseContext:
    """Inputs of a run plus everything the stages have produced so far."""

    df: pd.DataFrame
    feature_cols: list[str]
    spec: DatasetSpec
    config: TDAConfig
    X_norm: np.ndarray = field(init=False)
    record_ids: list[str] = field(init=False)
    group_cols: list[str] = field(init=False)
    results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X_norm = normalized_features(self.df, self.feature_cols)
        self.record_ids = self.spec.record_ids(self.df)
        self.group_cols = self.spec.present_group_cols(self.df)

    def to_base(self) -> dict[str, Any]:
        """The cacheable dict :func:`run_full_pipeline` consumes."""
        r = self.results
        labels = r["cluster_labels"]
        return {
            "X_norm":             self.X_norm,
            "record_ids":         self.record_ids,
            "feat_cols":          self.feature_cols,
            "cat_cols":           self.group_cols,
            "n_docs":             len(self.df),
            "n_clusters":         len(set(labels.tolist())) - (1 if -1 in labels else 0),
            "n_noise":            int((labels == -1).sum()),
            **r,
        }


Stage = Callable[[BaseContext], None]


def homology_stage(ctx: BaseContext) -> None:
    ctx.results["tda_info"] = compute_persistent_homology(ctx.X_norm, config=ctx.config)


def clustering_stage(ctx: BaseContext) -> None:
    labels, themes = compute_clusters(ctx.X_norm, ctx.df, ctx.record_ids, ctx.group_cols,
                                      list(ctx.spec.stat_cols), config=ctx.config)
    ctx.results["cluster_labels"] = labels
    ctx.results["themes"] = themes


def scoring_stage(ctx: BaseContext) -> None:
    anomalies, combined, iso, topo = detect_anomalies(
        ctx.X_norm, ctx.df, ctx.record_ids, ctx.results["cluster_labels"], ctx.feature_cols,
        ctx.group_cols, list(ctx.spec.display_cols), config=ctx.config)
    ctx.results.update({
        "anomalies": anomalies, "combined_scores": combined, "iso_scores": iso, "topo_scores": topo,
        "n_anomalies_total": int((np.asarray(combined) >= ctx.config.anomaly_high).sum()),
    })


def suspicious_stage(ctx: BaseContext) -> None:
    rules = list(ctx.spec.suspicious_rules) if ctx.spec.suspicious_rules is not None else None
    all_matches = build_suspicious_findings(
        ctx.df, ctx.results["combined_scores"], ctx.record_ids, top_k=None,
        iso_scores=ctx.results["iso_scores"], topo_scores=ctx.results["topo_scores"], rules=rules,
        group_cols=ctx.group_cols, display_cols=list(ctx.spec.display_cols), config=ctx.config)
    ctx.results["suspicious"] = all_matches[:ctx.config.suspicious_top_k]
    ctx.results["n_suspicious_total"] = len(all_matches)
    ctx.results["suspicious_ids"] = {str(f["sources"][0]) for f in all_matches if f.get("sources")}


def relationships_stage(ctx: BaseContext) -> None:
    basis = ctx.spec.relationship_basis(ctx.df) if ctx.spec.relationship_basis is not None else None
    rels, nodes, edges = compute_relationships(
        ctx.df, ctx.record_ids, ctx.results["cluster_labels"], ctx.group_cols,
        combined_scores=ctx.results["combined_scores"], suspicious_ids=ctx.results.pop("suspicious_ids"),
        basis_mask=basis, relationship_type=ctx.spec.relationship_type, config=ctx.config)
    ctx.results.update({"relationships": rels, "graph_nodes": nodes, "graph_edges": edges})


def drift_stage(ctx: BaseContext) -> None:
    ctx.results["drift"] = compute_drift(ctx.df, ctx.X_norm, ctx.results["cluster_labels"], ctx.feature_cols,
                                         ctx.spec.time_col, ctx.spec.metric_cols or None)


def topology_stage(ctx: BaseContext) -> None:
    ctx.results["topology"] = [
        make_finding("topology", lp["label"], lp["persistence"], lp.get("sources", []),
                     f"H₁ loop: birth={lp['birth']:.4f}, death={lp['death']:.4f}, "
                     f"persistence={lp['persistence']:.4f}.", lp)
        for lp in ctx.results["tda_info"].get("h1_features", [])
    ]


#: Execution order. Later stages read earlier results from the context.
DEFAULT_STAGES: tuple[Stage, ...] = (
    homology_stage, clustering_stage, scoring_stage, suspicious_stage,
    relationships_stage, drift_stage, topology_stage,
)


def check_abort(should_abort: Callable[[], bool] | None) -> None:
    if should_abort is not None and should_abort():
        raise PipelineSuperseded()


def run_stages(ctx: BaseContext, stages: tuple[Stage, ...] = DEFAULT_STAGES,
               should_abort: Callable[[], bool] | None = None) -> BaseContext:
    for stage in stages:
        check_abort(should_abort)
        stage(ctx)
        log.info("stage done: %s", stage.__name__)
    return ctx


def compute_base(
    df: pd.DataFrame,
    feature_cols: list[str],
    spec: DatasetSpec | None = None,
    config: TDAConfig | None = None,
    should_abort: Callable[[], bool] | None = None,
    stages: tuple[Stage, ...] = DEFAULT_STAGES,
) -> dict[str, Any]:
    """Every lens-independent stage, in order; ``should_abort`` is polled between them."""
    ctx = BaseContext(df, feature_cols, spec or DatasetSpec(), resolve(config))
    return run_stages(ctx, stages, should_abort).to_base()
