"""3-D Plotly figures: the plain 3-D graph and the 'galaxy' view with gas layers."""
from __future__ import annotations

import json

import networkx as nx
import numpy as np

from .graph import _build_nx, _hex_color, _neighbor_list, _norm_anom, _record_counts, _risk
from .layouts import _layout3d_disconnected, _normalize_pos3d, _positions_3d


def render_plotly_3d_json(nodes: list[dict], edges: list[dict]) -> dict:
    """Return the 3D Plotly figure as a JSON-serialisable dict for client-side rendering."""
    import plotly.graph_objects as go
    G = _build_nx(nodes, edges)
    if not G.number_of_nodes():
        return {"data": [], "layout": {}}
    pos3d = _positions_3d(G)
    anom = [G.nodes[n].get("avg_anomaly", 0.0) for n in G.nodes()]
    lo, hi = min(anom), max(anom)
    ex, ey, ez = [], [], []
    for u, v in G.edges():
        p0, p1 = pos3d[u], pos3d[v]
        ex += [p0[0], p1[0], None]; ey += [p0[1], p1[1], None]; ez += [p0[2], p1[2], None]
    edge_trace = go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                              line=dict(width=1, color="#3A4050"), hoverinfo="skip")
    nids = list(G.nodes())
    def _np3(n):
        a = G.nodes[n]
        return json.dumps({"id":n,"size":a.get("size",0),"n_members":a.get("n_members",a.get("size",0)),
            "avg_anomaly":a.get("avg_anomaly",0.0),"max_anomaly":a.get("max_anomaly",0.0),
            "risk_level":_risk(G, n),"n_neighbors":G.degree(n),
            "lens_center":a.get("lens_center",0.0),"lens_low":a.get("lens_low",0.0),
            "lens_high":a.get("lens_high",0.0),"label_top":a.get("label_top",{}),
            "date_range":a.get("date_range"),"feature_means":a.get("feature_means",{}),
            "members":(a.get("members") or [])[:8],"neighbors":_neighbor_list(G, n),
            "semantic_label":a.get("semantic_label",""),"short_label":a.get("short_label",""),
            "label_confidence":a.get("label_confidence"),"label_reasons":a.get("label_reasons",[]),"n_anomalous":a.get("n_anomalous"),"frac_anomalous":a.get("frac_anomalous"),"lift":a.get("lift")},
            separators=(',',':'))
    customdata = [[n, G.nodes[n].get("size",0), G.nodes[n].get("avg_anomaly",0.0),
                   G.nodes[n].get("max_anomaly",0.0), G.nodes[n].get("lens_center",0.0),
                   None, _np3(n)] for n in nids]
    n_nodes = G.number_of_nodes(); n_edges = G.number_of_edges()
    n_comps = nx.number_connected_components(G)
    _n_lbl3 = min(15, max(5, int(len(nids) ** 0.5 * 0.6)))
    _top3 = set(sorted(nids, key=lambda n: G.nodes[n].get("avg_anomaly", 0.0), reverse=True)[:_n_lbl3])
    def _slbl3(n):
        if n not in _top3:
            return ""
        a = G.nodes[n]
        sl = a.get("short_label")
        if sl:
            return ("⚠ " if _risk(G, n) == "HIGH" else "") + sl
        top = a.get("label_top") or {}
        fv = next((str(e[0]["value"])[:14] for e in top.values() if e), f"N{n}")
        return ("⚠ " if _risk(G, n) == "HIGH" else "") + fv
    node_trace = go.Scatter3d(
        x=[pos3d[n][0] for n in nids], y=[pos3d[n][1] for n in nids], z=[pos3d[n][2] for n in nids],
        mode="markers+text",
        text=[_slbl3(n) for n in nids],
        textposition="top center",
        textfont=dict(size=9, color="rgba(217,220,227,0.9)", family="IBM Plex Mono, monospace"),
        marker=dict(size=[max(5,min(28,G.nodes[n].get("size",1)**0.5*2.4)) for n in nids],
                    color=[_hex_color(_norm_anom(G.nodes[n].get("avg_anomaly",0.0),lo,hi)) for n in nids],
                    opacity=0.88, line=dict(width=1.5, color="rgba(15,17,23,0.7)"), symbol="circle"),
        customdata=customdata,
        hoverinfo="none",  # hide built-in tooltip; plotly_hover still fires for custom tooltip
    )
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=dict(text="TDA Mapper — 3D Shape of Your Data",
                   font=dict(color="#D9DCE3",size=15), x=0.5, xanchor="center"),
        scene=dict(
            bgcolor="#0F1117",
            dragmode="pan",
            xaxis=dict(showgrid=True,gridcolor="#1E2330",showticklabels=False,
                       title=dict(text="← Topological spread →",font=dict(color="#59616E",size=11)),
                       zeroline=False,backgroundcolor="#0F1117",showspikes=False),
            yaxis=dict(showgrid=True,gridcolor="#1E2330",showticklabels=False,
                       title=dict(text="← Topological spread →",font=dict(color="#59616E",size=11)),
                       zeroline=False,backgroundcolor="#0F1117",showspikes=False),
            zaxis=dict(showgrid=True,gridcolor="#2A2F3A",showticklabels=True,
                       title=dict(text="Filter height (how different from average)",
                                  font=dict(color="#878E9C",size=11)),
                       tickfont=dict(color="#59616E",size=9),
                       zeroline=True,zerolinecolor="#3A4050",backgroundcolor="#0F1117"),
            camera=dict(eye=dict(x=1.6,y=1.6,z=0.9), center=dict(x=0,y=0,z=0)),
            aspectmode="cube",
        ),
        paper_bgcolor="#0F1117",
        font=dict(color="#D9DCE3",family="IBM Plex Mono, monospace"),
        margin=dict(l=0,r=0,t=80,b=60),
        autosize=True, showlegend=False,
    ))
    out = json.loads(fig.to_json())
    out["stats"] = {"n_nodes": n_nodes, "n_edges": n_edges,
                    "n_components": n_comps, **_record_counts(nodes)}
    return out


# Galaxy framing: star shell as multiples of the graph's radius, and the camera
# position. Together they make the graph fill roughly half of the frame.
GALAXY_STAR_SHELL = (1.1, 1.8)
GALAXY_EYE = (0.76, 0.51, 0.40)          # ~28 % closer than a unit eye on the same direction


def render_galaxy_3d_json(nodes: list[dict], edges: list[dict]) -> dict:
    """
    Galaxy-style 3D Mapper visualization.

    Visual metaphor:
      Gold/warm core  = dense, low-anomaly clusters  (galactic centre)
      Teal/cyan arms  = medium-anomaly connected nodes (spiral arms)
      Red supernovas  = isolated high-anomaly outliers
      Starfield       = 420 background dots
      Gas layer       = translucent nebula around clusters + haze over each disk
      Glow halos      = transparent oversized markers behind each node
    """
    import plotly.graph_objects as go

    G = _build_nx(nodes, edges)
    if not G.number_of_nodes():
        return {"data": [], "layout": {}}

    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        pos3d = _layout3d_disconnected(G, comps)
    else:
        try:
            pos3d = nx.spring_layout(G, dim=3, seed=42, weight="weight",
                                     k=2.5 / max(1, G.number_of_nodes() ** 0.5))
        except Exception:
            pos3d = nx.spring_layout(G, dim=3, seed=42)
        pos3d = _normalize_pos3d(pos3d)

    # Z = lens centre, flattened to 40 % — creates the galaxy-disk shape
    lens_vals = [G.nodes[n].get("lens_center", 0.0) for n in G.nodes()]
    lmin, lmax = min(lens_vals), max(lens_vals)
    lrange = (lmax - lmin) or 1.0
    for n in G.nodes():
        raw_z = (G.nodes[n].get("lens_center", 0.0) - lmin) / lrange * 2 - 1
        pos3d[n] = np.array([pos3d[n][0], pos3d[n][1], raw_z * 0.4])

    anom = [G.nodes[n].get("avg_anomaly", 0.0) for n in G.nodes()]
    lo, hi = min(anom), max(anom)

    def _norm(v: float) -> float:
        return (v - lo) / (hi - lo + 1e-9) if hi > lo else 0.5

    def _gal_color(t: float) -> str:
        """Gold core → teal arm → supernova red."""
        if t < 0.4:
            r = int(255 * (1 - t / 0.4) + 0   * (t / 0.4))
            g = int(200 * (1 - t / 0.4) + 220 * (t / 0.4))
            b = int(50  * (1 - t / 0.4) + 200 * (t / 0.4))
        elif t < 0.72:
            t2 = (t - 0.4) / 0.32
            r = int(0   * (1 - t2) + 210 * t2)
            g = int(220 * (1 - t2) + 70  * t2)
            b = int(200 * (1 - t2) + 50  * t2)
        else:
            t2 = (t - 0.72) / 0.28
            r = int(210 * (1 - t2) + 255 * t2)
            g = int(70  * (1 - t2) + 30  * t2)
            b = int(50  * (1 - t2) + 20  * t2)
        return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

    nids = list(G.nodes())
    colors = [_gal_color(_norm(G.nodes[n].get("avg_anomaly", 0.0))) for n in nids]
    sizes  = [max(7, min(38, G.nodes[n].get("size", 1) ** 0.5 * 2.8)) for n in nids]

    # ── Starfield ─────────────────────────────────────────────────────────
    # The 3D box is fitted to everything drawn, so the star shell decides how
    # much of the view the graph gets. It is sized relative to the graph (not a
    # fixed 1.6–4.2): a shell that reached to 4.2 left the nodes about a third
    # of the frame, and every dataset had to be zoomed by hand.
    rng = np.random.default_rng(77)
    n_st = 420
    node_r = max(0.5, max(float(np.hypot(pos3d[n][0], pos3d[n][1])) for n in nids))
    r_s  = rng.uniform(GALAXY_STAR_SHELL[0] * node_r, GALAXY_STAR_SHELL[1] * node_r, n_st)
    th_s = rng.uniform(0, 2 * np.pi, n_st)
    ph_s = rng.uniform(0, np.pi, n_st)
    sx   = (r_s * np.sin(ph_s) * np.cos(th_s)).tolist()
    sy   = (r_s * np.sin(ph_s) * np.sin(th_s)).tolist()
    sz   = (r_s * np.cos(ph_s) * 0.18).tolist()
    sc   = [f"rgba(255,255,255,{round(rng.uniform(0.3,0.9),2)})" for _ in range(n_st)]
    ss   = rng.uniform(0.6, 2.2, n_st).tolist()

    star_trace = go.Scatter3d(x=sx, y=sy, z=sz, mode="markers",
        marker=dict(size=ss, color=sc, symbol="circle"),
        hoverinfo="skip", showlegend=False)

    # ── Gas layer (nebula) ─────────────────────────────────────────────────
    # Two clouds of large, almost transparent markers drawn behind everything:
    #   • a cool haze over each galaxy's disk (interstellar medium), and
    #   • a warm nebula around every star cluster, tinted with its colour and
    #     sized with its record count, so dense clusters glow the most.
    # hoverinfo="skip" keeps the gas out of Plotly's 3D picking, so clicks
    # still land on nodes and halos.
    gas_trace = _gas_layer(G, pos3d, nids, colors, sizes, comps)

    # ── Edges ──────────────────────────────────────────────────────────────
    ex, ey, ez = [], [], []
    for u, v in G.edges():
        p0, p1 = pos3d[u], pos3d[v]
        ex += [p0[0], p1[0], None]
        ey += [p0[1], p1[1], None]
        ez += [p0[2], p1[2], None]
    edge_trace = go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
        line=dict(width=0.8, color="rgba(120,160,220,0.25)"),
        hoverinfo="skip", showlegend=False)

    # ── Labels ──────────────────────────────────────────────────────────────
    _n_lbl = min(15, max(5, int(len(nids) ** 0.5 * 0.6)))
    _top_g = set(sorted(nids, key=lambda n: G.nodes[n].get("avg_anomaly", 0.0),
                        reverse=True)[:_n_lbl])

    def _slbl(n: int) -> str:
        if n not in _top_g: return ""
        a = G.nodes[n]
        sl = a.get("short_label")
        if sl:
            return ("⚠ " if _risk(G, n) == "HIGH" else "") + sl
        top = a.get("label_top") or {}
        fv = next((str(e[0]["value"])[:14] for e in top.values() if e), f"N{n}")
        return ("⚠ " if _risk(G, n) == "HIGH" else "") + fv

    # ── Customdata ──────────────────────────────────────────────────────────
    def _np(n: int) -> str:
        a = G.nodes[n]
        return json.dumps({"id":n,"size":a.get("size",0),
            "n_members":a.get("n_members",a.get("size",0)),
            "avg_anomaly":a.get("avg_anomaly",0.0),"max_anomaly":a.get("max_anomaly",0.0),
            "risk_level":_risk(G, n),"n_neighbors":G.degree(n),
            "lens_center":a.get("lens_center",0.0),"lens_low":a.get("lens_low",0.0),
            "lens_high":a.get("lens_high",0.0),"label_top":a.get("label_top",{}),
            "date_range":a.get("date_range"),"feature_means":a.get("feature_means",{}),
            "members":(a.get("members") or [])[:8],"neighbors":_neighbor_list(G, n),
            "semantic_label":a.get("semantic_label",""),"short_label":a.get("short_label",""),
            "label_confidence":a.get("label_confidence"),"label_reasons":a.get("label_reasons",[]),"n_anomalous":a.get("n_anomalous"),"frac_anomalous":a.get("frac_anomalous"),"lift":a.get("lift")},
            separators=(',',':'))

    customdata = [[n, G.nodes[n].get("size",0), G.nodes[n].get("avg_anomaly",0.0),
                   G.nodes[n].get("max_anomaly",0.0), G.nodes[n].get("lens_center",0.0),
                   None, _np(n)] for n in nids]

    # ── Glow halos (large transparent markers behind nodes) ────────────────
    # Plotly's 3D picking returns ONE object under the cursor and drops traces
    # with hoverinfo="skip". The halo is 2.2× the node and sits at the same
    # spot, so on small nodes it wins the pick and the click was swallowed
    # (no plotly_click at all). Giving the halo the node's customdata and
    # hoverinfo="none" makes a click on the glow count as a click on the node.
    halo_trace = go.Scatter3d(
        x=[pos3d[n][0] for n in nids],
        y=[pos3d[n][1] for n in nids],
        z=[pos3d[n][2] for n in nids],
        mode="markers",
        marker=dict(size=[s * 2.2 for s in sizes], color=colors, opacity=0.12,
                    line=dict(width=0)),
        customdata=customdata,
        hoverinfo="none", showlegend=False)

    n_nodes = G.number_of_nodes(); n_edges = G.number_of_edges()
    n_comps = nx.number_connected_components(G)

    node_trace = go.Scatter3d(
        x=[pos3d[n][0] for n in nids],
        y=[pos3d[n][1] for n in nids],
        z=[pos3d[n][2] for n in nids],
        mode="markers+text",
        text=[_slbl(n) for n in nids],
        textposition="top center",
        textfont=dict(size=9, color="rgba(255,240,200,0.95)", family="IBM Plex Mono, monospace"),
        marker=dict(size=sizes, color=colors, opacity=0.95,
                    line=dict(width=0.5, color="rgba(0,0,0,0.4)"), symbol="circle"),
        customdata=customdata,
        hoverinfo="none",  # hide built-in tooltip; plotly_hover still fires for custom tooltip
        showlegend=False,
    )

    fig = go.Figure(
        data=[star_trace, gas_trace, edge_trace, halo_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="TDA Galaxy — Data Universe",
                font=dict(color="#FFD580", size=16, family="IBM Plex Mono, monospace"),
                x=0.5, xanchor="center",
            ),
            scene=dict(
                bgcolor="#000008",
                dragmode="pan",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           backgroundcolor="#000008", showspikes=False,
                           title=dict(text="")),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           backgroundcolor="#000008", showspikes=False,
                           title=dict(text="")),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           backgroundcolor="#000008", showspikes=False,
                           title=dict(text="")),
                camera=dict(
                    eye=dict(zip("xyz", GALAXY_EYE)),
                    center=dict(x=0, y=0, z=0),
                    # no 'up' constraint → free orbit up/down and left/right
                ),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.6),
            ),
            # scene.dragmode="pan" handles 3D drag — layout.dragmode would affect 2D only
            paper_bgcolor="#000008",
            font=dict(color="#D9DCE3", family="IBM Plex Mono, monospace"),
            margin=dict(l=0, r=0, t=80, b=20),
            autosize=True,
            showlegend=False,
        ),
    )
    out = json.loads(fig.to_json())
    out["stats"] = {"n_nodes": n_nodes, "n_edges": n_edges,
                    "n_components": n_comps, **_record_counts(nodes)}
    return out


def _gas_layer(G: nx.Graph, pos3d: dict, nids: list, colors: list[str], sizes: list[float],
               comps: list):
    """
    The nebula behind the galaxy view: one Scatter3d of soft, translucent blobs.

    Disk haze:  ~600 bluish particles spread over the components (share ∝ node
                count), flattened like the disk, so each galaxy sits in a glow.
    Cluster gas: per node, 4–18 particles (∝ √records) in a Gaussian cloud whose
                radius grows with the node, tinted with the node colour.
    Arm gas:    thin streaks along every edge (particles jittered around the
                line between the two nodes, colour blended from both ends), so
                linked clusters read as one arm instead of separate blobs.
    Plotly's WebGL scatter ignores per-point alpha, so the softness comes from
    the trace-level marker opacity; overlapping blobs add up to the glow.
    """
    import plotly.graph_objects as go

    rng = np.random.default_rng(7)
    xs, ys, zs, cs, ss = [], [], [], [], []

    def _rgba(hex_or_rgba: str, _alpha: float) -> str:
        h = hex_or_rgba.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        # lift towards white so the gas reads as glow, not as a darker node
        r, g, b = (int(r + (255 - r) * 0.1), int(g + (255 - g) * 0.1), int(b + (255 - b) * 0.1))
        return f"rgb({r},{g},{b})"

    # disk haze per component
    n_total = max(1, G.number_of_nodes())
    for comp in comps:
        members = [n for n in comp if n in pos3d]
        if not members:
            continue
        pts = np.array([pos3d[n] for n in members])
        centre = pts.mean(axis=0)
        spread = float(np.sqrt(((pts[:, :2] - centre[:2]) ** 2).sum(axis=1).mean())) if len(pts) > 1 else 0.0
        radius = max(0.18, spread * 1.6)
        k = max(4, int(320 * len(members) / n_total))
        r = radius * np.sqrt(rng.uniform(0, 1, k))
        th = rng.uniform(0, 2 * np.pi, k)
        xs += (centre[0] + r * np.cos(th)).tolist()
        ys += (centre[1] + r * np.sin(th)).tolist()
        zs += (centre[2] + rng.normal(0, radius * 0.12, k)).tolist()
        cs += [f"rgb({rng.integers(60, 100)},{rng.integers(80, 130)},{rng.integers(190, 235)})"
               for _ in range(k)]
        ss += rng.uniform(22, 46, k).tolist()

    # nebula around each node
    for n, col, size in zip(nids, colors, sizes):
        p = pos3d[n]
        k = int(max(4, min(18, size * 0.5)))
        sigma = 0.035 + size * 0.003
        xs += (p[0] + rng.normal(0, sigma, k)).tolist()
        ys += (p[1] + rng.normal(0, sigma, k)).tolist()
        zs += (p[2] + rng.normal(0, sigma * 0.5, k)).tolist()
        cs += [_rgba(col, 0.0) for _ in range(k)]
        ss += rng.uniform(10, 30, k).tolist()

    # streaks along the connections (spiral arms)
    def _rgb_of(col: str):
        h = col.lstrip("#")
        return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=float)

    col_of = {n: _rgb_of(c) for n, c in zip(nids, colors)}
    for u, v in G.edges():
        p0, p1 = pos3d[u], pos3d[v]
        length = float(np.linalg.norm(p1 - p0))
        k = int(max(4, min(24, length * 40)))
        t = rng.uniform(0.08, 0.92, k)
        pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        pts += rng.normal(0, 0.025, pts.shape) * np.array([1, 1, 0.5])
        xs += pts[:, 0].tolist(); ys += pts[:, 1].tolist(); zs += pts[:, 2].tolist()
        for ti in t:
            c = col_of[u] * (1 - ti) + col_of[v] * ti
            cs.append(f"rgb({int(c[0])},{int(c[1])},{int(c[2])})")
        ss += rng.uniform(8, 20, k).tolist()

    return go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", name="gas",
                        marker=dict(size=ss, color=cs, opacity=0.05, symbol="circle",
                                    line=dict(width=0)),
                        hoverinfo="skip", showlegend=False)

