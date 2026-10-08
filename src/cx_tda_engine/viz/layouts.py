"""Node positions: 2-D networkx layouts (incl. disconnected-component tiling) and 3-D positions."""
from __future__ import annotations

import networkx as nx
import numpy as np

from ..core.risk import mapper_risk_counts
from .graph import _build_nx, _neighbor_list, _risk


def _layout(G: nx.Graph, name: str) -> dict:
    comps = list(nx.connected_components(G))
    is_disconnected = len(comps) > 1

    # Shell gets its own concentric-ring implementation that is visually distinct
    # from the grid tiling used for the other layouts.
    if name == "shell":
        return _shell_layout(G, comps)

    if is_disconnected and name in ("kamada_kawai", "spectral"):
        return _layout_disconnected(G, comps, name)

    fns = {
        "spring":       lambda: nx.spring_layout(G, seed=42, weight="weight",
                                                  k=2.0 / max(1, G.number_of_nodes() ** 0.5)),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G),
        "shell":        lambda: nx.shell_layout(G),
        "spectral":     lambda: nx.spectral_layout(G) if G.number_of_nodes() > 2 else nx.spring_layout(G, seed=42),
        "circular":     lambda: nx.circular_layout(G),
    }
    try:
        return _normalize_pos(fns.get(name, fns["spring"])())
    except Exception:
        return _normalize_pos(nx.spring_layout(G, seed=42))


def _shell_layout(G: nx.Graph, comps: list) -> dict:
    """
    True concentric-ring layout that reveals hub vs. peripheral structure.

    Ring 0 (innermost): nodes in the largest multi-node components — the
        topology hubs that are most connected to other groups.
    Ring 1: nodes in smaller multi-node components.
    Ring 2+: isolated singleton nodes spread evenly in the outer ring.

    This makes Shell visually distinct from Kamada-Kawai:
      Kamada-Kawai → grid of components, preserving topological distance
      Shell         → concentric rings, showing centrality / connectivity class
    """
    import math

    multi  = sorted([c for c in comps if len(c) > 1], key=len, reverse=True)
    single = [list(c)[0] for c in comps if len(c) == 1]

    pos: dict = {}

    # ── Inner ring(s): multi-node components ─────────────────────────────────
    # Lay out each component internally, then place its centroid on a ring whose
    # radius grows with the number of multi-node components.
    if multi:
        n_multi = len(multi)
        # Radius of the outermost multi-component ring — scales with count
        max_r = max(0.35, min(0.65, 0.15 + 0.05 * math.sqrt(n_multi)))
        # Use up to 3 rings for multi-node components
        n_rings = max(1, min(3, math.ceil(math.sqrt(n_multi / 3))))
        per_ring = math.ceil(n_multi / n_rings)

        for idx, comp in enumerate(multi):
            ring  = idx // per_ring
            slot  = idx %  per_ring
            r     = max_r * (ring + 1) / n_rings
            n_slots = min(per_ring, n_multi - ring * per_ring)
            angle = 2 * math.pi * slot / max(1, n_slots)
            cx, cy = r * math.cos(angle), r * math.sin(angle)

            # Internal layout of this component
            sub = G.subgraph(comp).copy()
            try:
                sub_pos = nx.spring_layout(sub, seed=42, scale=0.08) if len(sub) > 1 else {list(sub.nodes())[0]: np.zeros(2)}
            except Exception:
                sub_pos = {n: np.zeros(2) for n in sub.nodes()}

            for node, p in sub_pos.items():
                pos[node] = np.array([cx + p[0], cy + p[1]])

    # ── Outer ring: isolated singleton nodes ─────────────────────────────────
    if single:
        outer_r = 0.90 if multi else 0.70
        for i, node in enumerate(single):
            angle = 2 * math.pi * i / len(single)
            pos[node] = np.array([outer_r * math.cos(angle),
                                   outer_r * math.sin(angle)])

    return _normalize_pos(pos) if pos else {}


def _layout_disconnected(G: nx.Graph, comps: list, name: str) -> dict:
    """Lay out each connected component separately, pack them into a grid,
    then normalise all coordinates to [-1, 1] so the viewport always fits."""
    import math
    comps = sorted(comps, key=len, reverse=True)
    n_cols = max(1, math.ceil(math.sqrt(len(comps))))
    raw: dict = {}
    col, row = 0, 0
    # Cell size scales with sqrt(component_count) so large grids stay compact
    spacing = max(2.5, 10.0 / max(1, math.sqrt(len(comps))))

    for comp in comps:
        sub = G.subgraph(comp).copy()
        try:
            if name == "kamada_kawai":
                sub_pos = (nx.kamada_kawai_layout(sub)
                           if len(sub) > 1
                           else {list(sub.nodes())[0]: np.zeros(2)})
            else:  # spectral
                sub_pos = (nx.spectral_layout(sub)
                           if len(sub) > 2
                           else nx.spring_layout(sub, seed=42))
        except Exception:
            sub_pos = nx.spring_layout(sub, seed=42)

        ox, oy = col * spacing, -row * spacing
        for node, p in sub_pos.items():
            raw[node] = p + np.array([ox, oy])

        col += 1
        if col >= n_cols:
            col = 0
            row += 1

    # Normalise to [-1, 1] so Plotly/Matplotlib viewport always shows everything
    return _normalize_pos(raw)


def _normalize_pos(pos: dict) -> dict:
    """Rescale layout positions to the [-1, 1] square."""
    if not pos:
        return pos
    xs = [float(p[0]) for p in pos.values()]
    ys = [float(p[1]) for p in pos.values()]
    xr = (max(xs) - min(xs)) or 1.0
    yr = (max(ys) - min(ys)) or 1.0
    scale = max(xr, yr)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return {n: np.array([(p[0] - cx) / scale * 2, (p[1] - cy) / scale * 2])
            for n, p in pos.items()}


def positions_3d(nodes: list[dict], edges: list[dict]) -> dict:
    """Node id → (x, y, z) exactly as the TDA Mapper 3D view lays it out: spring layout
    ("topological spread") for x / y, the lens value ("filter height") for z, all in [-1, 1]."""
    G = _build_nx(nodes, edges)
    if not G.number_of_nodes():
        return {}
    return _positions_3d(G)


def _positions_3d(G: nx.Graph) -> dict:
    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        pos3d = _layout3d_disconnected(G, comps)
    else:
        try:
            pos3d = nx.spring_layout(G, dim=3, seed=42, weight="weight",
                                     k=2.0 / max(1, G.number_of_nodes() ** 0.5))
        except Exception:
            pos3d = nx.spring_layout(G, dim=3, seed=42)
        pos3d = _normalize_pos3d(pos3d)
    lens_vals = [G.nodes[n].get("lens_center", 0.0) for n in G.nodes()]
    lmin, lmax = min(lens_vals), max(lens_vals)
    lrange = (lmax - lmin) or 1.0
    for n in G.nodes():
        raw_z = (G.nodes[n].get("lens_center", 0.0) - lmin) / lrange * 2 - 1
        pos3d[n] = np.array([pos3d[n][0], pos3d[n][1], raw_z])
    return pos3d


def _layout3d_disconnected(G: nx.Graph, comps: list) -> dict:
    """3D layout for disconnected graphs: per-component spring, then grid-tile."""
    import math
    comps = sorted(comps, key=len, reverse=True)
    n_cols = max(1, math.ceil(math.sqrt(len(comps))))
    spacing = max(2.5, 10.0 / max(1, math.sqrt(len(comps))))
    raw: dict = {}
    col, row = 0, 0
    for comp in comps:
        sub = G.subgraph(comp).copy()
        try:
            sub_pos = (nx.spring_layout(sub, dim=3, seed=42)
                       if len(sub) > 1
                       else {list(sub.nodes())[0]: np.zeros(3)})
        except Exception:
            sub_pos = {n: np.zeros(3) for n in sub.nodes()}
        ox, oy = col * spacing, -row * spacing
        for n, p in sub_pos.items():
            raw[n] = p + np.array([ox, oy, 0.0])
        col += 1
        if col >= n_cols:
            col = 0; row += 1
    return _normalize_pos3d(raw)


def _normalize_pos3d(pos: dict) -> dict:
    """Rescale 3D layout positions to the [-1,1] cube."""
    if not pos:
        return pos
    arr = np.array([p for p in pos.values()])
    center = arr.mean(axis=0)
    scale = (arr - center).std() * 2 or 1.0
    return {n: (np.array(p) - center) / scale for n, p in pos.items()}


def render_layout_json(nodes: list[dict], edges: list[dict], layout: str = "spring") -> dict:
    """Node positions + node payloads as plain JSON.

    The backend owns the layout computation; the browser owns all rendering
    (no HTML/JS is generated here). Positions are normalised to [-1, 1].
    """
    G = _build_nx(nodes, edges)
    pos = _layout(G, layout) if G.number_of_nodes() else {}
    out_nodes = []
    for n in G.nodes():
        a = G.nodes[n]
        x, y = pos.get(n, (0.0, 0.0))
        out_nodes.append({
            "id": n, "x": float(x), "y": float(y),
            "size": a.get("size", 0), "n_members": a.get("n_members", a.get("size", 0)),
            "avg_anomaly": a.get("avg_anomaly", 0.0), "max_anomaly": a.get("max_anomaly", 0.0),
            "risk_level": _risk(G, n), "n_neighbors": G.degree(n),
            "lens_center": a.get("lens_center", 0.0), "lens_low": a.get("lens_low", 0.0),
            "lens_high": a.get("lens_high", 0.0), "label_top": a.get("label_top", {}),
            "date_range": a.get("date_range"), "feature_means": a.get("feature_means", {}),
            "members": (a.get("members") or [])[:8], "neighbors": _neighbor_list(G, n),
            "semantic_label": a.get("semantic_label", ""), "short_label": a.get("short_label", ""),
            "label_confidence": a.get("label_confidence"), "label_reasons": a.get("label_reasons", []),
            "n_anomalous": a.get("n_anomalous"), "frac_anomalous": a.get("frac_anomalous"), "lift": a.get("lift"),
        })
    out_edges = [{"source": u, "target": v, "weight": G.edges[u, v].get("weight", 1)}
                 for u, v in G.edges()]
    stats = {"n_nodes": len(out_nodes), "n_edges": len(out_edges), **mapper_risk_counts(out_nodes)}
    return {"layout": layout, "nodes": out_nodes, "edges": out_edges, "stats": stats}

