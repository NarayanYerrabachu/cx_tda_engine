from cx_tda_engine import (
    TDAConfig,
    graph_node_state,
    loop_class,
    mapper_node_risk_level,
    mapper_risk_counts,
    mapper_risk_from_lift,
    node_risk_level,
    priority,
)


def test_graph_node_state_is_relative_to_dataset_rate():
    assert graph_node_state(0.05, 0.16) == "normal"       # below the dataset rate
    assert graph_node_state(0.21, 0.16) == "warning"      # ~1.3x
    assert graph_node_state(0.40, 0.16) == "anomalous"    # 2.5x
    assert graph_node_state(0.90, 0.0) == "normal"        # no anomalies anywhere


def test_mapper_tiers_follow_the_same_rule():
    base = 0.16
    assert mapper_node_risk_level(0.40, base) == "HIGH"
    assert mapper_node_risk_level(0.21, base) == "WATCH"
    assert mapper_node_risk_level(0.10, base) == "LOW"
    assert mapper_node_risk_level(1.0, 0.0) == "LOW"
    assert mapper_risk_from_lift(None) == "LOW" and mapper_risk_from_lift(2.0) == "HIGH"
    assert node_risk_level({"risk_level": "WATCH"}) == "WATCH"
    assert node_risk_level({"lift": 3.0}) == "HIGH"
    assert node_risk_level({}) == "LOW"


def test_tiers_respect_config():
    strict = TDAConfig(high_lift=5.0, watch_lift=3.0)
    assert mapper_node_risk_level(0.40, 0.16, strict) == "LOW"
    assert mapper_risk_from_lift(3.5, strict) == "WATCH"


def test_risk_counts():
    c = mapper_risk_counts([{"risk_level": "HIGH", "size": 3}, {"risk_level": "WATCH", "size": 10},
                            {"lift": 0.5, "size": 100}])
    assert c == {"n_high": 1, "n_watch": 1, "n_low": 1,
                 "records_high": 3, "records_watch": 10, "records_low": 100, "records_total": 113}


def test_loop_class_buckets():
    assert loop_class(None) is None
    assert loop_class(0) == "acyclic"
    assert loop_class(1) == "weak"
    assert loop_class(5) == "weak"
    assert loop_class(6) == "moderate"
    assert loop_class(50) == "moderate"
    assert loop_class(51) == "high"
    assert loop_class(3, TDAConfig(loop_weak_max=2, loop_moderate_max=4)) == "moderate"


def test_priority_labels():
    assert priority(0.95) == "HIGH" and priority(0.5) == "MEDIUM" and priority(0.1) == "REVIEW"
