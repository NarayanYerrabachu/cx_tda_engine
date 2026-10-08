import numpy as np
import pytest

from cx_tda_engine import (
    DEFAULT_CONFIG,
    TDAConfig,
    build_mapper_graph,
    dbscan_capped,
    graph_node_state,
    mapper_node_risk_level,
    run_mapper,
)


def test_graph_nodes_scored_over_all_members():
    """Node scores use every member's score, not the 20-id `sources` sample."""
    n = 200
    X = np.zeros((n, 2))
    X[:, 0] = np.linspace(0, 1, n)
    lens = X[:, 0].copy()
    scores = np.zeros(n)
    scores[150:] = 0.9
    g = build_mapper_graph(X, lens, n_intervals=2, overlap=0.0, record_ids=[f"R{i}" for i in range(n)],
                           anomaly_scores=scores)
    nodes = g["nodes"]
    assert nodes
    base = float((scores >= DEFAULT_CONFIG.anomaly_high).mean())
    for nd in nodes:
        assert len(nd["sources"]) <= 20 < nd["size"]
        assert nd["state"] == graph_node_state(nd["frac_anomalous"], base)
        assert nd["id"].startswith("node-")
    upper = max(nodes, key=lambda nd: nd["lens_mean"])
    assert upper["frac_anomalous"] == pytest.approx(0.5, abs=0.02)
    assert upper["state"] == "anomalous"
    lower = min(nodes, key=lambda nd: nd["lens_mean"])
    assert lower["avg_anomaly"] == pytest.approx(0.0, abs=1e-6) and lower["state"] == "normal"


def test_build_mapper_graph_two_blobs_connected_within_and_not_across():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (150, 2)), rng.normal(0, 0.1, (150, 2)) + [10, 0]])
    g = build_mapper_graph(X, X[:, 0], n_intervals=10, overlap=0.5)
    members = [set(int(s) for s in nd["sources"]) for nd in g["nodes"]]
    left = {i for i, m in enumerate(members) if all(s < 150 for s in m)}
    right = {i for i, m in enumerate(members) if all(s >= 150 for s in m)}
    assert left and right and left | right == set(range(len(members)))
    for e in g["edges"]:
        assert (e["source"] in left) == (e["target"] in left)
        assert e["weight"] >= 1


def test_build_mapper_graph_fit_cap_keeps_membership_complete():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    g = build_mapper_graph(X, X[:, 0], n_intervals=4, overlap=0.0, config=TDAConfig(mapper_fit_cap=50))
    assert sum(nd["size"] for nd in g["nodes"]) == 600


def test_run_mapper_tiers_are_relative_to_dataset_rate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    scores = np.where(X[:, 1] > 1.0, 0.8, 0.2)
    res = run_mapper(X, X[:, 0], anomaly_scores=scores, record_ids=[str(i) for i in range(300)])
    st, nodes = res["stats"], res["nodes"]
    assert nodes
    thr = DEFAULT_CONFIG.anomaly_high
    assert st["anomaly_threshold"] == thr
    assert st["anomaly_base_rate"] == pytest.approx(float((scores >= thr).mean()), abs=1e-4)
    for n in nodes:
        assert n["n_anomalous"] == sum(scores[int(m)] >= thr for m in n["members"])
        assert n["frac_anomalous"] == pytest.approx(n["n_anomalous"] / n["size"], abs=1e-4)
        assert n["lift"] == pytest.approx(n["frac_anomalous"] / st["anomaly_base_rate"], abs=2e-3)
        assert n["risk_level"] == mapper_node_risk_level(n["frac_anomalous"], st["anomaly_base_rate"])
    assert all(n["risk_level"] == "LOW" for n in nodes if n["n_anomalous"] == 0)
    assert all(n["risk_level"] == "HIGH" for n in nodes if n["frac_anomalous"] == 1.0)
    assert st["records_high"] + st["records_watch"] + st["records_low"] == st["records_total"] == sum(
        n["size"] for n in nodes)


def test_run_mapper_outputs_neighbors_histogram_bands():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    res = run_mapper(X, X[:, 0], n_intervals=8, overlap=0.4, anomaly_scores=rng.random(300))
    nodes, st = res["nodes"], res["stats"]
    assert nodes and st["n_high"] + st["n_watch"] + st["n_low"] == st["n_nodes"]
    assert st["dbscan_eps"] > 0
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        assert n["risk_level"] in ("HIGH", "WATCH", "LOW")
        for nb in n["neighbors"]:
            assert nb["risk_level"] == by_id[nb["id"]]["risk_level"]
            assert nb["shared"] >= 1
    assert len(res["interval_bands"]) == 8
    assert sum(b["count"] for b in res["lens_histogram"]) <= 300
    assert all(e["source"] != e["target"] for e in res["edges"])


def test_run_mapper_empty_input():
    assert run_mapper(np.zeros((0, 3)), np.zeros(0))["nodes"] == []


def test_dbscan_capped_assigns_every_row():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.2, (2500, 2)), rng.normal(0, 0.2, (2500, 2)) + [5, 5]])
    full = dbscan_capped(X, 0.5, 5, seed=0)
    capped = dbscan_capped(X, 0.5, 5, seed=0, config=TDAConfig(run_mapper_fit_cap=500))
    assert len(capped) == len(full) == 5000
    assert len(set(full.tolist()) - {-1}) == len(set(capped.tolist()) - {-1}) == 2
    # the two blobs stay separate under the capped fit
    assert len(set(capped[:2500].tolist()) - {-1}) == 1 and len(set(capped[2500:].tolist()) - {-1}) == 1
