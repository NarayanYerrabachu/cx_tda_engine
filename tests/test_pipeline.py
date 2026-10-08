import json

import numpy as np
import pytest

from cx_tda_engine import (
    DatasetSpec,
    InMemoryBaseCache,
    PipelineSuperseded,
    TDAConfig,
    cached_base,
    clear_base_cache,
    loop_class,
    run_full_pipeline,
    seed_base_cache,
)

from .conftest import FEATURES, make_df

TOP_LEVEL = {"meta", "themes", "anomalies", "suspicious", "relationships", "drift", "topology", "graph",
             "documents", "record_ids", "lens_values", "cluster_labels", "combined_scores"}


def test_result_contract(df, spec):
    res = run_full_pipeline(df, FEATURES, lens_name="pca", spec=spec)
    assert set(res) == TOP_LEVEL
    meta = res["meta"]
    assert meta["n_docs"] == len(df) and meta["data_type"] == "tabular" and meta["lens"] == "pca"
    assert meta["n_anomalies"] == len(res["anomalies"]) <= meta["n_anomalies_total"]
    assert meta["n_suspicious"] == len(res["suspicious"]) <= meta["n_suspicious_total"]
    assert meta["n_relationships"] == len(res["relationships"])
    assert meta["n_drift"] == len(res["drift"]) and meta["n_topology"] == len(res["topology"])
    assert meta["numeric_features"] == FEATURES and meta["category_features"] == ["group_a", "group_b"]
    assert meta["anomaly_threshold"] == 0.6 and meta["seed"] == 42
    assert meta["mapper_n_intervals"] == 15 and meta["mapper_overlap"] == 0.5
    assert meta["mapper_requested"] == {"n_intervals": 10, "overlap": 0.5}
    assert len(res["record_ids"]) == len(res["cluster_labels"]) == len(res["lens_values"]) == len(df)
    assert res["record_ids"][0] == "R0000"
    assert {"nodes", "edges", "mapper_nodes", "mapper_edges"} == set(res["graph"])
    assert all("lens_mean" in n and "state" in n for n in res["graph"]["mapper_nodes"])
    json.dumps(res)   # the whole thing is JSON-serialisable


def test_meta_tda_block(df, spec):
    tda = run_full_pipeline(df, FEATURES, spec=spec)["meta"]["tda"]
    assert tda["available"] is True
    assert tda["loop_class"] == loop_class(tda["betti_1"])


def test_purely_numeric_defaults(df):
    res = run_full_pipeline(df, FEATURES)
    assert res["record_ids"] == [str(i) for i in range(len(df))]
    assert res["relationships"] == [] and res["meta"]["category_features"] == []
    assert res["anomalies"][0]["title"] == res["anomalies"][0]["sources"][0]


@pytest.mark.parametrize("lens", ["pca", "density", "eccentricity", "feature", "umap"])
def test_all_lenses(df, spec, lens):
    res = run_full_pipeline(df, FEATURES, lens_name=lens, spec=spec)
    assert res["graph"]["mapper_nodes"]


def test_errors_are_returned_not_raised(df):
    assert "error" in run_full_pipeline(df, ["nope"])
    assert "error" in run_full_pipeline(df.head(3), FEATURES)
    assert "error" in run_full_pipeline(df, FEATURES, config=TDAConfig(min_rows=1000))


def test_lens_change_reuses_cached_base(df, spec):
    first = run_full_pipeline(df, FEATURES, lens_name="pca", spec=spec)
    second = run_full_pipeline(df, FEATURES, lens_name="eccentricity", spec=spec)
    assert first["meta"]["cached_base"] is False and second["meta"]["cached_base"] is True
    assert first["anomalies"] == second["anomalies"]
    assert first["combined_scores"] == second["combined_scores"]
    assert first["lens_values"] != second["lens_values"]
    assert cached_base(df, FEATURES) is not None
    clear_base_cache()
    assert cached_base(df, FEATURES) is None


def test_private_cache_and_seeding(df):
    cache = InMemoryBaseCache()
    res = run_full_pipeline(df, FEATURES, cache=cache)
    assert len(cache) == 1 and cached_base(df, FEATURES) is None
    seed_base_cache(df, FEATURES, cache.get(df, FEATURES))
    assert run_full_pipeline(df, FEATURES)["meta"]["cached_base"] is True
    assert run_full_pipeline(df, FEATURES, use_cache=False)["meta"]["cached_base"] is False
    assert res["meta"]["cached_base"] is False


def test_abort_raises_and_leaves_cache_empty(df):
    calls = {"n": 0}

    def abort():
        calls["n"] += 1
        return calls["n"] >= 2

    with pytest.raises(PipelineSuperseded):
        run_full_pipeline(df, FEATURES, should_abort=abort)
    assert cached_base(df, FEATURES) is None


def test_domain_hooks(df, spec):
    hooked = DatasetSpec(
        id_col=spec.id_col, group_cols=spec.group_cols, time_col="year",
        metric_cols={"metric": "Metric"},
        suspicious_rules=(lambda row, score: ["metric high"] if row["metric"] > 1.2 else [],),
        relationship_basis=lambda d: (d["metric"] > 0).to_numpy(), relationship_type="positive_metric",
    )
    res = run_full_pipeline(df, FEATURES, spec=hooked)
    assert res["meta"]["n_suspicious_total"] == int((df["metric"] > 1.2).sum())
    assert all(r["extra"]["type"] == "positive_metric" for r in res["relationships"])
    assert res["drift"][0]["extra"]["feature"] == "metric"


def test_determinism_across_runs(spec):
    a = run_full_pipeline(make_df(seed=3), FEATURES, spec=spec, use_cache=False)
    b = run_full_pipeline(make_df(seed=3), FEATURES, spec=spec, use_cache=False)
    assert a["combined_scores"] == b["combined_scores"]
    assert a["cluster_labels"] == b["cluster_labels"]
    assert [n["sources"] for n in a["graph"]["mapper_nodes"]] == [n["sources"] for n in b["graph"]["mapper_nodes"]]


def test_config_changes_results(df, spec):
    base = run_full_pipeline(df, FEATURES, spec=spec, use_cache=False)
    strict = run_full_pipeline(df, FEATURES, spec=spec, use_cache=False,
                               config=TDAConfig(anomaly_high=0.9, anomaly_top_k=5))
    assert strict["meta"]["n_anomalies"] == 5
    assert strict["meta"]["n_anomalies_total"] < base["meta"]["n_anomalies_total"]
    assert strict["meta"]["anomaly_threshold"] == 0.9
    assert np.array(strict["combined_scores"]).max() <= 1.0
