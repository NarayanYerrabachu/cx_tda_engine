"""Tunable parameters of the TDA engine.

Every number that used to be an environment variable or a schema constant in
the CortXplorer demo lives here, so a caller can run two pipelines with
different settings in one process and nothing is read from ``os.environ``
unless :meth:`TDAConfig.from_env` is called explicitly.

Thresholds are expressed as *lifts* relative to the dataset (how much more
anomalous a group is than the data as a whole) rather than absolute cut-offs:
with full per-record scores almost every large group contains one high-scoring
record and has an average near the dataset mean, so absolute rules flag
everything. See :mod:`cx_tda_engine.risk` for the classification rules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from typing import Any

_ENV_PREFIX = "TDA_"

# Field name → environment variable suffix. Kept explicit so the demo's
# existing ``TDA_*`` variables map one-to-one onto config fields.
_ENV_NAMES: dict[str, str] = {
    "seed": "SEED",
    "sample_n": "SAMPLE_N",
    "contamination": "CONTAMINATION",
    "dbscan_eps": "DBSCAN_EPS",
    "dbscan_min_samples": "DBSCAN_MIN_SAMPLES",
    "n_jobs": "N_JOBS",
    "mapper_fit_cap": "MAPPER_FIT_CAP",
    "anomaly_top_k": "ANOMALY_TOP_K",
    "suspicious_top_k": "SUSPICIOUS_TOP_K",
    "run_mapper_fit_cap": "RUN_MAPPER_FIT_CAP",
    "mapper_max_pairs": "MAPPER_MAX_PAIRS",
}


@dataclass(frozen=True)
class TDAConfig:
    """All knobs of the pipeline. Immutable; derive variants with :meth:`with_`."""

    # ── Reproducibility and compute budget ────────────────────────────────────
    seed: int = 42
    #: Max points fed to ripser (persistent homology is O(n^3) in memory).
    sample_n: int = 3000
    #: IsolationForest contamination, clipped to [0.005, 0.10] at use.
    contamination: float = 0.03
    #: Global DBSCAN eps. When left at this default, eps is auto-tuned from
    #: k-NN distances (see :func:`cx_tda_engine.clustering.auto_eps`).
    dbscan_eps: float = 0.9
    dbscan_min_samples: int = 5
    #: sklearn worker count. joblib's fork backend duplicates the parent's
    #: virtual memory per worker; ``-1`` on a 16-CPU host has OOM-killed a
    #: container mid-pipeline, so the default is 1.
    n_jobs: int = 1
    #: Max rows for a single interval's DBSCAN fit in :func:`build_mapper_graph`.
    #: Above this the fit runs on a seeded subsample and the rest are assigned by
    #: nearest centroid (DBSCAN's neighbour graph is quadratic in dense regions).
    mapper_fit_cap: int = 4000
    #: Fixed per-interval fit cap for :func:`run_mapper`; 0 = estimate memory instead.
    run_mapper_fit_cap: int = 0
    #: Neighbour-pair budget for one :func:`run_mapper` interval (~1 GB at 60 M).
    mapper_max_pairs: float = 60_000_000.0
    #: Minimum rows for a meaningful run.
    min_rows: int = 5

    # ── Display caps (the true totals are reported separately in meta) ────────
    anomaly_top_k: int = 30
    suspicious_top_k: int = 20

    # ── Classification thresholds ─────────────────────────────────────────────
    #: A record is "anomalous" when its combined score >= this.
    anomaly_high: float = 0.60
    anomaly_med: float = 0.40
    #: Node/group tiers from lift = frac_anomalous / dataset base rate.
    high_lift: float = 2.0
    watch_lift: float = 1.25
    #: node_score = w * frac_anomalous + (1 - w) * avg_anomaly
    score_frac_weight: float = 0.5
    #: beta_1 buckets: 0 acyclic, 1..weak_max weak, ..moderate_max moderate, else high.
    loop_weak_max: int = 5
    loop_moderate_max: int = 50

    # ── Mapper resolution floors (too-coarse covers collapse to one node) ─────
    mapper_min_n_intervals: int = 15
    mapper_min_overlap: float = 0.4

    #: Score blend in :func:`detect_anomalies`: combined = iso_w * iso + (1 - iso_w) * topo.
    iso_weight: float = 0.6

    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.anomaly_high <= 1.0:
            raise ValueError("anomaly_high must be in (0, 1]")
        if self.watch_lift > self.high_lift:
            raise ValueError("watch_lift must not exceed high_lift")
        if self.loop_weak_max > self.loop_moderate_max:
            raise ValueError("loop_weak_max must not exceed loop_moderate_max")
        if self.min_rows < 2:
            raise ValueError("min_rows must be at least 2")

    @property
    def effective_contamination(self) -> float:
        return min(0.10, max(0.005, self.contamination))

    def with_(self, **changes: Any) -> "TDAConfig":
        """A copy with the given fields replaced."""
        return replace(self, **changes)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **overrides: Any) -> "TDAConfig":
        """Build a config from ``TDA_*`` environment variables (the demo's convention).

        Unset variables keep the dataclass defaults; ``overrides`` win over both.
        """
        env = os.environ if environ is None else environ
        types = {f.name: f.type for f in fields(cls)}
        values: dict[str, Any] = {}
        for name, suffix in _ENV_NAMES.items():
            raw = env.get(_ENV_PREFIX + suffix)
            if raw is None:
                continue
            kind = types[name]
            values[name] = float(raw) if kind == "float" else int(raw)
        values.update(overrides)
        return cls(**values)


DEFAULT_CONFIG = TDAConfig()


def resolve(config: TDAConfig | None) -> TDAConfig:
    return DEFAULT_CONFIG if config is None else config
