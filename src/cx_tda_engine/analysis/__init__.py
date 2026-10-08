"""Lens-independent analysis stages. Each module is one concern with one public entry point."""
from .anomalies import anomaly_scores, detect_anomalies
from .clustering import auto_eps, compute_clusters
from .drift import cluster_drift, compute_drift, temporal_feature_drift, temporal_metric_drift
from .homology import compute_persistent_homology, homology_projection
from .relationships import compute_relationships
from .suspicious import build_suspicious_findings, score_rule

__all__ = [
    "anomaly_scores", "detect_anomalies",
    "auto_eps", "compute_clusters",
    "cluster_drift", "compute_drift", "temporal_feature_drift", "temporal_metric_drift",
    "compute_persistent_homology", "homology_projection",
    "compute_relationships",
    "build_suspicious_findings", "score_rule",
]
