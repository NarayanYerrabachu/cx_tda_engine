import pytest

from cx_tda_engine import DEFAULT_CONFIG, TDAConfig


def test_defaults_match_demo_values():
    c = DEFAULT_CONFIG
    assert (c.seed, c.sample_n, c.contamination, c.dbscan_eps, c.dbscan_min_samples) == (42, 3000, 0.03, 0.9, 5)
    assert (c.n_jobs, c.mapper_fit_cap, c.anomaly_top_k, c.suspicious_top_k) == (1, 4000, 30, 20)
    assert (c.anomaly_high, c.anomaly_med, c.high_lift, c.watch_lift, c.score_frac_weight) == (0.6, 0.4, 2.0, 1.25, 0.5)
    assert (c.loop_weak_max, c.loop_moderate_max) == (5, 50)
    assert (c.mapper_min_n_intervals, c.mapper_min_overlap) == (15, 0.4)


def test_from_env_reads_tda_variables_and_overrides():
    env = {"TDA_SEED": "7", "TDA_CONTAMINATION": "0.05", "TDA_ANOMALY_TOP_K": "10", "UNRELATED": "x"}
    c = TDAConfig.from_env(env, anomaly_high=0.7)
    assert (c.seed, c.contamination, c.anomaly_top_k, c.anomaly_high) == (7, 0.05, 10, 0.7)
    assert c.sample_n == 3000           # untouched default


def test_from_env_ignores_real_environment_when_mapping_given(monkeypatch):
    monkeypatch.setenv("TDA_SEED", "99")
    assert TDAConfig.from_env({}).seed == 42
    assert TDAConfig.from_env().seed == 99


def test_with_returns_modified_copy():
    base = TDAConfig()
    c = base.with_(anomaly_top_k=5)
    assert c.anomaly_top_k == 5 and base.anomaly_top_k == 30


def test_contamination_is_clipped():
    assert TDAConfig(contamination=0.5).effective_contamination == 0.10
    assert TDAConfig(contamination=0.0).effective_contamination == 0.005


@pytest.mark.parametrize("bad", [dict(anomaly_high=0.0), dict(watch_lift=3.0), dict(loop_weak_max=60), dict(min_rows=1)])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        TDAConfig(**bad)
