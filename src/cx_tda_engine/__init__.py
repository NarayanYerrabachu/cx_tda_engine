"""cx_tda_engine: topological data analysis for tabular data.

Quickstart::

    import pandas as pd
    from cx_tda_engine import DatasetSpec, TDAConfig, run_full_pipeline

    df = pd.read_csv("records.csv")
    result = run_full_pipeline(
        df, ["amount", "duration", "score"], lens_name="pca",
        spec=DatasetSpec(id_col="record_id", group_cols=("region", "category"), time_col="year"),
        config=TDAConfig(anomaly_top_k=50),
    )
    result["anomalies"][0]["title"], result["graph"]["mapper_nodes"]
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .anomalies import anomaly_scores, detect_anomalies
from .clustering import auto_eps, compute_clusters
from .config import DEFAULT_CONFIG, TDAConfig
from .drift import cluster_drift, compute_drift, temporal_feature_drift, temporal_metric_drift
from .findings import build_suspicious_findings, score_rule
from .homology import compute_persistent_homology, homology_projection
from .lenses import LENS_REGISTRY, BaseLens, get_lens, lens_catalog, register_lens
from .mapper import build_mapper_graph, dbscan_capped, run_mapper
from .pipeline import (
    DEFAULT_CACHE,
    BaseCache,
    DatasetSpec,
    PipelineSuperseded,
    cached_base,
    clear_base_cache,
    compute_base,
    run_full_pipeline,
    seed_base_cache,
)
from .preprocessing import normalize, normalized_features
from .relationships import compute_relationships
from .risk import (
    graph_node_state,
    loop_class,
    mapper_node_risk_level,
    mapper_risk_counts,
    mapper_risk_from_lift,
    node_risk_level,
    priority,
)
from .text import (
    TEXT_SPEC,
    Document,
    DocumentProcessor,
    chunk_documents,
    detect_processor,
    documents_from_texts,
    documents_to_dataframe,
    register_processor,
    run_text_pipeline,
)
from .types import Finding, HomologyResult, MapperEdge, MapperNode, PipelineResult, SuspiciousRule

try:
    __version__ = _pkg_version("cx_tda_engine")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0+src"

__all__ = [
    "__version__",
    "TDAConfig", "DEFAULT_CONFIG", "DatasetSpec",
    "run_full_pipeline", "compute_base", "PipelineSuperseded",
    "BaseCache", "DEFAULT_CACHE", "clear_base_cache", "cached_base", "seed_base_cache",
    "normalize", "normalized_features",
    "homology_projection", "compute_persistent_homology",
    "BaseLens", "LENS_REGISTRY", "get_lens", "register_lens", "lens_catalog",
    "build_mapper_graph", "run_mapper", "dbscan_capped",
    "anomaly_scores", "detect_anomalies",
    "auto_eps", "compute_clusters",
    "compute_relationships",
    "compute_drift", "cluster_drift", "temporal_feature_drift", "temporal_metric_drift",
    "build_suspicious_findings", "score_rule",
    "graph_node_state", "mapper_node_risk_level", "mapper_risk_from_lift", "node_risk_level",
    "mapper_risk_counts", "loop_class", "priority",
    "Document", "DocumentProcessor", "TEXT_SPEC", "documents_from_texts", "chunk_documents",
    "documents_to_dataframe", "run_text_pipeline", "register_processor", "detect_processor",
    "Finding", "MapperNode", "MapperEdge", "HomologyResult", "PipelineResult", "SuspiciousRule",
]
