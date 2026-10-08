"""Result contract of the pipeline, as TypedDicts.

These describe the dictionaries :func:`cx_tda_engine.run_full_pipeline` returns.
They are plain dicts at runtime (JSON-serialisable, so a web service can hand
them to a frontend unchanged); the TypedDicts exist for readers and type checkers.
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

import pandas as pd


class Finding(TypedDict):
    """One entry of the ``anomalies`` / ``suspicious`` / ``themes`` / ``relationships``
    / ``drift`` / ``topology`` lists. ``kind`` names the list it came from."""

    kind: str
    title: str
    score: float
    sources: list[str]
    detail: str
    extra: dict[str, Any]


class MapperNode(TypedDict, total=False):
    """A node of the pipeline Mapper graph (``graph.mapper_nodes``)."""

    id: str
    size: int
    sources: list[str]
    interval: int
    cluster: int
    lens_mean: float
    avg_anomaly: float
    max_anomaly: float
    frac_anomalous: float
    n_anomalous: int
    node_score: float
    lift: float | None
    state: str


class MapperEdge(TypedDict):
    source: int
    target: int
    weight: int


class HomologyResult(TypedDict, total=False):
    available: bool
    betti_0: int
    betti_1: int
    betti_0_raw: int
    betti_1_raw: int
    noise_threshold: float
    median_h1_persistence: float
    max_persistence: float
    loop_class: str | None
    persistence_ratio: float | None
    h0_features: list[dict[str, Any]]
    h1_features: list[dict[str, Any]]
    diagram_h0: list[dict[str, Any]]
    diagram_h1: list[dict[str, Any]]
    diagram_max: float


class PipelineGraph(TypedDict):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    mapper_nodes: list[MapperNode]
    mapper_edges: list[MapperEdge]


class PipelineResult(TypedDict, total=False):
    meta: dict[str, Any]
    themes: list[Finding]
    anomalies: list[Finding]
    suspicious: list[Finding]
    relationships: list[Finding]
    drift: list[Finding]
    topology: list[Finding]
    graph: PipelineGraph
    documents: list[Any]
    record_ids: list[str]
    lens_values: list[float]
    cluster_labels: list[int]
    combined_scores: list[float]
    error: str


def make_finding(kind: str, title: str, score: float, sources: list[str], detail: str,
                 extra: dict[str, Any] | None = None) -> Finding:
    """The one place a finding dict is assembled, so every list has the same shape."""
    return {"kind": kind, "title": title, "score": round(float(score), 4), "sources": list(sources),
            "detail": detail, "extra": dict(extra or {})}


#: A suspicious-record rule: given one row and its combined anomaly score, return
#: the reasons (human-readable strings) it should be flagged for, or an empty list.
SuspiciousRule = Callable[[pd.Series, float], list[str]]
