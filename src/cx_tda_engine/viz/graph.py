"""Shared helpers over the Mapper node/edge dicts: networkx construction, counts, labels, colours."""
from __future__ import annotations

import networkx as nx

from ..core.risk import node_risk_level


def _record_counts(nodes: list[dict]) -> dict[str, int]:
    """Records vs. group memberships of a Mapper graph.

    Mapper intervals overlap, so one record usually sits in several groups: the
    sum of the group sizes ("memberships") is larger than the number of records.
    ``total_records`` is the number of distinct records; without member lists
    only the memberships are known and both are reported as that sum.
    """
    memberships = int(sum(n.get("size", 0) for n in nodes))
    members = [n.get("members") for n in nodes]
    if nodes and all(m is not None for m in members):
        distinct: set = set()
        for m in members:
            distinct.update(m or [])
        return {"total_records": len(distinct), "memberships": memberships}
    return {"total_records": memberships, "memberships": memberships}


def _build_nx(nodes: list[dict], edges: list[dict]) -> nx.Graph:
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "members"})
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e.get("weight", 1))
    return G


def _neighbor_list(G, n: int, limit: int = 6) -> list[dict]:
    """Return a list of neighbor dicts for node n, sorted by shared members."""
    result = []
    for nb in G.neighbors(n):
        w = G.edges[n, nb].get("weight", 1)
        result.append({
            "id":      nb,
            "shared":  w,
            "size":    G.nodes[nb].get("size", 0),
            "anomaly": round(G.nodes[nb].get("avg_anomaly", 0.0), 4),
            "risk_level": _risk(G, nb),
            "short_label": G.nodes[nb].get("short_label", ""),
            "semantic_label": G.nodes[nb].get("semantic_label", ""),
        })
    result.sort(key=lambda x: x["shared"], reverse=True)
    return result[:limit]


def _risk(G, n) -> str:
    """Backend risk tier of node n (set by run_mapper; derived if missing)."""
    return node_risk_level(G.nodes[n])


def _node_name(G, n) -> str:
    """Unique, readable node name used in every tooltip: "<short label> · Node n".

    Semantic labels repeat across nodes (e.g. several "COVID shock Crash"), so the
    node id is always included to keep each reference unique.
    """
    lbl = G.nodes[n].get("short_label") or G.nodes[n].get("semantic_label")
    return f"{lbl} · Node {n}" if lbl else f"Node {n}"


def _fmt_num(v: float, key: str = "") -> str:
    """Format a number compactly. Skips K/M/B for years, codes, IDs."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    kl = key.lower()
    is_plain = any(k in kl for k in ("year", "jahr", "code", "id", "nr", "number",
                                      "count", "anzahl", "index", "seq"))
    if is_plain:
        return str(int(v)) if v == int(v) else f"{v:.2f}"
    usd = abs(v) >= 1e4 and any(k in kl for k in (
        "usd", "budget", "cost", "revenue", "amount", "kapital", "markt",
        "eur", "price", "value", "overran", "loss", "profit", "income"))
    prefix = "$" if usd else ""
    if abs(v) >= 1e9: return f"{prefix}{v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"{prefix}{v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"{prefix}{v/1e3:.1f}K"
    return f"{prefix}{v:.4f}" if abs(v) < 1 else f"{prefix}{v:.2f}"


def _hex_color(t: float) -> str:
    """Map t ∈ [0,1] → teal → purple → red hex."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t * 2
        r, g, b = (int(0x5E + (0x9B - 0x5E) * s),
                   int(0x9C + (0x7F - 0x9C) * s),
                   int(0xA6 + (0xD4 - 0xA6) * s))
    else:
        s = (t - 0.5) * 2
        r, g, b = (int(0x9B + (0xE0 - 0x9B) * s),
                   int(0x7F + (0x52 - 0x7F) * s),
                   int(0xD4 + (0x52 - 0xD4) * s))
    return f"#{r:02X}{g:02X}{b:02X}"


def _norm_anom(val: float, lo: float, hi: float) -> float:
    return (val - lo) / (hi - lo + 1e-9)

