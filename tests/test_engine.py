import numpy as np

from cx_tda_engine import DatasetSpec, TDAConfig, TDAEngine, cached_base
from cx_tda_engine.pipeline import DEFAULT_STAGES, BaseContext, run_stages

from .conftest import FEATURES


def test_engine_runs_with_own_cache_and_config(df, spec):
    engine = TDAEngine(config=TDAConfig(anomaly_top_k=7), spec=spec)
    first = engine.run(df, FEATURES, lens_name="pca")
    second = engine.run(df, FEATURES, lens_name="density")
    assert first["meta"]["n_anomalies"] == 7
    assert first["meta"]["cached_base"] is False and second["meta"]["cached_base"] is True
    assert cached_base(df, FEATURES) is None          # the default cache was not touched
    engine.clear_cache()
    assert engine.run(df, FEATURES)["meta"]["cached_base"] is False


def test_engine_mapper_from_dataframe(df, spec):
    engine = TDAEngine(spec=spec)
    res = engine.run(df, FEATURES)
    graph = engine.mapper(df, FEATURES, lens_name="pca", anomaly_scores=np.array(res["combined_scores"]))
    assert graph["stats"]["n_nodes"] > 0
    assert all(m.startswith("R") for n in graph["nodes"] for m in n["members"])


def test_stage_chain_is_extensible(df):
    seen = []

    def custom_stage(ctx: BaseContext) -> None:
        seen.append(sorted(ctx.results))
        ctx.results["custom"] = len(ctx.record_ids)

    ctx = BaseContext(df, FEATURES, DatasetSpec(), TDAConfig())
    base = run_stages(ctx, DEFAULT_STAGES + (custom_stage,)).to_base()
    assert base["custom"] == len(df)
    assert "topology" in seen[0] and "cluster_labels" in seen[0]
    assert {"X_norm", "record_ids", "anomalies", "relationships", "drift", "tda_info"} <= set(base)
