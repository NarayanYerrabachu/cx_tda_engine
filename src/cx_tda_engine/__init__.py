"""cx_tda_engine: topological data analysis for tabular and unstructured data.

Layers (each a subpackage, dependencies point downwards only)::

    adapters   text and other inputs -> (DataFrame, feature columns)
    pipeline   DatasetSpec, stage chain, cache, run_full_pipeline, TDAEngine
    mapper     Mapper graphs (pipeline and exploration variants)
    analysis   homology, clustering, anomaly scores, relationships, drift, suspicious
    lenses     Strategy + registry of filter functions
    core       result types, risk rules, preprocessing
    config     TDAConfig (all numeric knobs)
    viz        optional renderers over Mapper output

Quickstart::

    import pandas as pd
    from cx_tda_engine import DatasetSpec, TDAConfig, run_full_pipeline

    df = pd.read_csv("records.csv")
    result = run_full_pipeline(
        df, ["amount", "duration", "score"], lens_name="pca",
        spec=DatasetSpec(id_col="record_id", group_cols=("region", "category"), time_col="year"),
        config=TDAConfig(anomaly_top_k=50),
    )
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .adapters import (
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
from .analysis import (
    anomaly_scores,
    auto_eps,
    build_suspicious_findings,
    cluster_drift,
    compute_clusters,
    compute_drift,
    compute_persistent_homology,
    compute_relationships,
    detect_anomalies,
    homology_projection,
    score_rule,
    temporal_feature_drift,
    temporal_metric_drift,
)
from .config import DEFAULT_CONFIG, TDAConfig
from .core import (
    Finding,
    HomologyResult,
    MapperEdge,
    MapperNode,
    PipelineResult,
    SuspiciousRule,
    graph_node_state,
    loop_class,
    make_finding,
    mapper_node_risk_level,
    mapper_risk_counts,
    mapper_risk_from_lift,
    node_risk_level,
    normalize,
    normalized_features,
    priority,
)
from .lenses import LENS_REGISTRY, BaseLens, get_lens, lens_catalog, register_lens
from .mapper import build_mapper_graph, dbscan_capped, run_mapper
from .pipeline import (
    DEFAULT_CACHE,
    BaseCache,
    DatasetSpec,
    InMemoryBaseCache,
    PipelineSuperseded,
    TDAEngine,
    cached_base,
    clear_base_cache,
    compute_base,
    run_full_pipeline,
    seed_base_cache,
)

try:
    __version__ = _pkg_version("cx_tda_engine")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0+src"

__all__ = [
    "__version__",
    # configuration and entry points
    "TDAConfig", "DEFAULT_CONFIG", "DatasetSpec", "TDAEngine",
    "run_full_pipeline", "compute_base", "PipelineSuperseded",
    "BaseCache", "InMemoryBaseCache", "DEFAULT_CACHE", "clear_base_cache", "cached_base", "seed_base_cache",
    # core
    "normalize", "normalized_features", "make_finding",
    "graph_node_state", "mapper_node_risk_level", "mapper_risk_from_lift", "node_risk_level",
    "mapper_risk_counts", "loop_class", "priority",
    "Finding", "MapperNode", "MapperEdge", "HomologyResult", "PipelineResult", "SuspiciousRule",
    # lenses and mapper
    "BaseLens", "LENS_REGISTRY", "get_lens", "register_lens", "lens_catalog",
    "build_mapper_graph", "run_mapper", "dbscan_capped",
    # analysis
    "homology_projection", "compute_persistent_homology",
    "anomaly_scores", "detect_anomalies", "auto_eps", "compute_clusters", "compute_relationships",
    "compute_drift", "cluster_drift", "temporal_feature_drift", "temporal_metric_drift",
    "build_suspicious_findings", "score_rule",
    # adapters
    "Document", "DocumentProcessor", "TEXT_SPEC", "documents_from_texts", "chunk_documents",
    "documents_to_dataframe", "run_text_pipeline", "register_processor", "detect_processor",
]
