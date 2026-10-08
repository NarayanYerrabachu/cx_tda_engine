"""Classification rules shared by every view of the pipeline output.

All rules are *relative to the dataset*: a Mapper group is HIGH when its share
of anomalous members is at least ``high_lift`` times the dataset-wide share.
Fixed cut-offs on the average score were abandoned because small outlier
fragments always exceeded them while the dataset's own baseline group landed on
WATCH, which made the tiers useless for prioritising.
"""
from __future__ import annotations

from typing import Any

from .config import TDAConfig, resolve

NODE_STATES = ("anomalous", "warning", "normal")
RISK_LEVELS = ("HIGH", "WATCH", "LOW")
_STATE_TO_TIER = {"anomalous": "HIGH", "warning": "WATCH", "normal": "LOW"}


def lift(frac_anomalous: float | None, base_rate: float | None) -> float | None:
    """Enrichment of a group vs. the dataset; None when nothing is anomalous anywhere."""
    if not base_rate:
        return None
    return float(frac_anomalous or 0.0) / float(base_rate)


def graph_node_state(frac_anomalous: float, base_rate: float, config: TDAConfig | None = None) -> str:
    """"anomalous" / "warning" / "normal" for a pipeline Mapper node."""
    cfg = resolve(config)
    lf = lift(frac_anomalous, base_rate)
    if lf is None:
        return "normal"
    if lf >= cfg.high_lift:
        return "anomalous"
    if lf >= cfg.watch_lift:
        return "warning"
    return "normal"


def mapper_node_risk_level(frac_anomalous: float | None, base_rate: float | None,
                           config: TDAConfig | None = None) -> str:
    """HIGH / WATCH / LOW for a Mapper group from its anomalous share vs. the dataset's."""
    return _STATE_TO_TIER[graph_node_state(float(frac_anomalous or 0.0), float(base_rate or 0.0), config)]


def mapper_risk_from_lift(lift_value: float | None, config: TDAConfig | None = None) -> str:
    """Tier from an already computed lift (None → LOW: nothing anomalous to compare with)."""
    cfg = resolve(config)
    if lift_value is None:
        return "LOW"
    if lift_value >= cfg.high_lift:
        return "HIGH"
    if lift_value >= cfg.watch_lift:
        return "WATCH"
    return "LOW"


def node_risk_level(node: dict[str, Any], config: TDAConfig | None = None) -> str:
    """Tier of a Mapper node / neighbour payload. :func:`run_mapper` always sets
    ``risk_level``; payloads that only carry ``lift`` (or neither) are derived."""
    return str(node.get("risk_level") or mapper_risk_from_lift(node.get("lift"), config))


def mapper_risk_counts(nodes: list[dict[str, Any]], config: TDAConfig | None = None) -> dict[str, int]:
    """Groups and record memberships per risk tier.

    Returns n_high / n_watch / n_low (groups) and records_high / records_watch /
    records_low / records_total (sum of group sizes, i.e. memberships, since
    Mapper groups overlap). The record figures make "125 HIGH groups"
    interpretable: they may hold 2 % of the data.
    """
    out = {"n_high": 0, "n_watch": 0, "n_low": 0,
           "records_high": 0, "records_watch": 0, "records_low": 0, "records_total": 0}
    for n in nodes:
        tier = node_risk_level(n, config).lower()
        size = int(n.get("size") or n.get("n_members") or 0)
        out[f"n_{tier}"] += 1
        out[f"records_{tier}"] += size
        out["records_total"] += size
    return out


def loop_class(betti_1: int | None, config: TDAConfig | None = None) -> str | None:
    """Bucket beta_1 (loops above the noise band) into acyclic / weak / moderate / high.

    Descriptive only; nothing here is significance-tested.
    """
    if betti_1 is None:
        return None
    cfg = resolve(config)
    b = int(betti_1)
    if b <= 0:
        return "acyclic"
    if b <= cfg.loop_weak_max:
        return "weak"
    if b <= cfg.loop_moderate_max:
        return "moderate"
    return "high"


def priority(score: float, config: TDAConfig | None = None) -> str:
    """HIGH / MEDIUM / REVIEW label for one record's combined anomaly score."""
    cfg = resolve(config)
    if score >= cfg.anomaly_high:
        return "HIGH"
    if score >= cfg.anomaly_med:
        return "MEDIUM"
    return "REVIEW"
