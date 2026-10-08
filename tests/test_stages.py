"""Lens-independent stages: normalisation, homology, clustering, anomalies, relationships, drift, suspicious."""
import numpy as np
import pytest

from cx_tda_engine import (
    DEFAULT_CONFIG,
    TDAConfig,
    auto_eps,
    build_suspicious_findings,
    compute_clusters,
    compute_drift,
    compute_persistent_homology,
    compute_relationships,
    detect_anomalies,
    loop_class,
    normalize,
)
from cx_tda_engine.analysis.drift import cluster_drift, temporal_metric_drift

from .conftest import FEATURES


def test_normalize_shape_nan_free_zero_mean(df):
    X = df[FEATURES].to_numpy(dtype=float, copy=True)
    X[0, 0] = np.nan
    X[1, 1] = np.inf
    Xn = normalize(X)
    assert Xn.shape == X.shape and not np.isnan(Xn).any()
    np.testing.assert_allclose(Xn.mean(axis=0), 0.0, atol=1e-10)


def test_homology_reports_betti_at_noise_scale(X_norm):
    h = compute_persistent_homology(X_norm)
    assert h["available"] is True
    assert h["betti_0_raw"] == len(X_norm)            # raw H0 bars == sample size
    assert h["betti_0"] <= h["betti_0_raw"]
    assert h["betti_1"] <= h["betti_1_raw"]
    assert h["loop_class"] == loop_class(h["betti_1"])
    if h["noise_threshold"] > 0:
        assert h["persistence_ratio"] == pytest.approx(h["max_persistence"] / h["noise_threshold"], rel=1e-2)
    assert all(f["label"] for f in h["h1_features"])
    assert len(h["diagram_h1"]) >= len(h["h1_features"])


def test_homology_honours_sample_n(X_norm):
    h = compute_persistent_homology(X_norm, sample_n=50)
    assert h["betti_0_raw"] == 50


def test_homology_degrades_without_ripser(X_norm, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "ripser":
            raise ImportError("no ripser")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    h = compute_persistent_homology(X_norm)
    assert h["available"] is False and h["loop_class"] is None


def test_auto_eps_never_returns_zero():
    X = np.zeros((40, 3))
    X[:4, 0] = 1.0
    assert auto_eps(X) > 0
    with pytest.raises(ValueError, match="same feature values"):
        auto_eps(np.zeros((37, 3)))


def test_compute_clusters_labels_and_themes(df, X_norm, record_ids):
    labels, themes = compute_clusters(X_norm, df, record_ids, ["group_a", "group_b"], ["metric"])
    assert len(labels) == len(df)
    assert set(labels.tolist()) <= set(range(-1, len(df)))
    for t in themes:
        assert t["kind"] == "theme" and "cluster_id" in t["extra"]
        assert "metric" in t["extra"]["stats"]
        assert t["title"].startswith(f"Cluster {t['extra']['cluster_id']}")


def test_compute_clusters_without_groups(df, X_norm, record_ids):
    labels, themes = compute_clusters(X_norm, df, record_ids)
    assert len(labels) == len(df)
    assert all(t["title"] == f"Cluster {t['extra']['cluster_id']}" for t in themes)


def test_compute_clusters_tolerates_empty_group_column(df, X_norm, record_ids):
    """A grouping column that is all-missing (e.g. parent_file of plain-text documents)
    must not crash the theme labels."""
    df = df.copy()
    df["empty"] = None
    labels, themes = compute_clusters(X_norm, df, record_ids, ["empty", "group_a"])
    assert len(labels) == len(df) and themes
    assert all("None" not in t["title"] and "nan" not in t["title"] for t in themes)


def test_detect_anomalies(df, X_norm, record_ids):
    labels, _ = compute_clusters(X_norm, df, record_ids)
    findings, scores, iso, topo = detect_anomalies(X_norm, df, record_ids, labels, FEATURES,
                                                   ["group_a"], ["metric"], top_k=10)
    assert len(findings) == 10 and len(scores) == len(iso) == len(topo) == len(df)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert [f["score"] for f in findings] == sorted((f["score"] for f in findings), reverse=True)
    # the planted outlier block dominates the top findings
    assert sum(f["sources"][0] in set(record_ids[:15]) for f in findings) >= 8
    for f in findings:
        assert f["kind"] == "anomaly"
        assert {"iso_score", "topo_score", "flagged_by", "priority", "metric"} <= set(f["extra"])
        assert len(f["extra"]["flagged_by"]) == 3
        assert f["title"].startswith(f["sources"][0] + " — ")


def test_anomaly_scores_are_deterministic(df, X_norm, record_ids):
    labels, _ = compute_clusters(X_norm, df, record_ids)
    _, a, _, _ = detect_anomalies(X_norm, df, record_ids, labels, FEATURES)
    _, b, _, _ = detect_anomalies(X_norm, df, record_ids, labels, FEATURES)
    np.testing.assert_array_equal(a, b)
    _, c, _, _ = detect_anomalies(X_norm, df, record_ids, labels, FEATURES, config=TDAConfig(seed=1))
    assert not np.array_equal(a, c)


def test_relationships_full_counts(df, X_norm, record_ids):
    labels, _ = compute_clusters(X_norm, df, record_ids)
    scores = np.linspace(0, 1, len(df))
    susp = set(record_ids[::3])
    rels, nodes, edges = compute_relationships(df, record_ids, labels, ["group_a", "group_b"],
                                               combined_scores=scores, suspicious_ids=susp)
    assert rels and nodes and edges
    assert [r["extra"]["weight"] for r in rels] == sorted((r["extra"]["weight"] for r in rels), reverse=True)
    for r in rels:
        ex = r["extra"]
        grp = df[(df["group_a"].astype(str) == ex["a"]) & (df["group_b"].astype(str) == ex["b"])]
        assert ex["weight"] == len(grp)
        assert ex["n_anomalous"] == int((scores[grp.index] >= DEFAULT_CONFIG.anomaly_high).sum())
        assert ex["n_suspicious"] == sum(record_ids[i] in susp for i in grp.index)
        assert ex["a_stats"]["n_records"] == int((df["group_a"].astype(str) == ex["a"]).sum())
        assert ex["type"] == "cooccurrence"


def test_relationships_with_basis_mask(df, X_norm, record_ids):
    labels, _ = compute_clusters(X_norm, df, record_ids)
    mask = (df["metric"] > 0.3).to_numpy()
    rels, _, _ = compute_relationships(df, record_ids, labels, ["group_a", "group_b"], basis_mask=mask,
                                       relationship_type="high_metric")
    sub = df[mask]
    for r in rels:
        ex = r["extra"]
        assert ex["weight"] == int(((sub["group_a"] == ex["a"]) & (sub["group_b"] == ex["b"])).sum())
        assert ex["type"] == "high_metric"


def test_relationships_need_two_group_columns(df, X_norm, record_ids):
    assert compute_relationships(df, record_ids, np.zeros(len(df), int), ["group_a"]) == ([], [], [])


def test_drift_prefers_time_then_clusters(df, X_norm):
    labels = np.zeros(len(df), dtype=int)
    by_time = compute_drift(df, X_norm, labels, FEATURES, time_col="year")
    assert all(d["kind"] == "drift" and d["extra"]["mid_year"] is not None for d in by_time)
    none = compute_drift(df, X_norm, labels, FEATURES)      # one cluster, no time: nothing to compare
    assert none == []
    labels[: len(df) // 2] = 1
    by_cluster = cluster_drift(X_norm, FEATURES, labels)
    assert all(d["extra"]["mid_year"] is None for d in by_cluster)
    for d in by_time + by_cluster:
        assert {"feature", "early_mean", "late_mean", "delta", "direction"} <= set(d["extra"])


def test_metric_drift_uses_raw_values(df):
    out = temporal_metric_drift(df, "year", {"metric": "Metric"})
    assert out and out[0]["title"].startswith("Metric drift")
    assert out[0]["extra"]["feature"] == "metric"


def test_suspicious_default_rule_and_custom_rules(df, record_ids):
    scores = np.linspace(0, 1, len(df))
    default = build_suspicious_findings(df, scores, record_ids, top_k=None, group_cols=["group_a"])
    assert len(default) == int((scores >= 0.6).sum())
    assert all(f["extra"]["priority"] == "HIGH" for f in default)
    assert default[0]["score"] >= default[-1]["score"]

    def big_metric(row, score):
        return ["Metric above 1"] if row["metric"] > 1.0 else []

    custom = build_suspicious_findings(df, scores, record_ids, top_k=None, rules=[big_metric],
                                       display_cols=["metric"])
    assert len(custom) == int((df["metric"] > 1.0).sum())
    assert all(f["extra"]["reasons"] == ["Metric above 1"] and "metric" in f["extra"] for f in custom)
    assert len(build_suspicious_findings(df, scores, record_ids, top_k=3)) == 3
