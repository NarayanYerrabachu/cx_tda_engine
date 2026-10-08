"""2-D Plotly figure of a Mapper graph (as a JSON-ready dict)."""
from __future__ import annotations

import json

from .graph import _build_nx, _hex_color, _neighbor_list, _norm_anom, _risk
from .layouts import _layout


def render_plotly_json(nodes: list[dict], edges: list[dict], layout: str = "spring") -> dict:
    """Return the 2D Plotly figure as a JSON-serialisable dict for client-side rendering."""
    import plotly.graph_objects as go
    # Build the same figure as render_plotly but return fig.to_dict() instead of HTML
    # so the frontend can call Plotly.newPlot() directly in the parent DOM.
    G = _build_nx(nodes, edges)
    if not G.number_of_nodes():
        return {"data": [], "layout": {}}
    pos = _layout(G, layout)
    anom = [G.nodes[n].get("avg_anomaly", 0.0) for n in G.nodes()]
    lo, hi = min(anom), max(anom)
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(width=0.8, color="#3A4050"), hoverinfo="skip")
    nids = list(G.nodes())
    def _np(n):
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
                   None, _np(n)] for n in nids]
    _n_lblj = min(15, max(5, int(len(nids) ** 0.5 * 0.6)))
    _topj = set(sorted(nids, key=lambda n: G.nodes[n].get("avg_anomaly", 0.0), reverse=True)[:_n_lblj])
    def _slblj(n):
        if n not in _topj:
            return ""
        a = G.nodes[n]
        sl = a.get("short_label")
        if sl:
            return ("⚠ " if _risk(G, n) == "HIGH" else "") + sl
        top = a.get("label_top") or {}
        fv = next((str(e[0]["value"])[:14] for e in top.values() if e), f"N{n}")
        return ("⚠ " if _risk(G, n) == "HIGH" else "") + fv
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nids], y=[pos[n][1] for n in nids],
        mode="markers+text",
        text=[_slblj(n) for n in nids],
        textposition="top center",
        textfont=dict(size=9, color="rgba(217,220,227,0.9)", family="IBM Plex Mono, monospace"),
        marker=dict(size=[max(6,min(40,G.nodes[n].get("size",1)**0.5*3.5)) for n in nids],
                    color=[_hex_color(_norm_anom(G.nodes[n].get("avg_anomaly",0.0),lo,hi)) for n in nids],
                    line=dict(width=1,color="#1A1D24")),
        customdata=customdata,
        hoverinfo="none",  # hide built-in tooltip; plotly_hover still fires for custom tooltip
    )
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=dict(text=f"TDA Mapper — {layout.replace('_',' ').title()} layout",
                   font=dict(color="#D9DCE3",size=14)),
        showlegend=False, paper_bgcolor="#13161E", plot_bgcolor="#13161E",
        font=dict(color="#D9DCE3",family="IBM Plex Mono, monospace"),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,autorange=True),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,autorange=True),
        margin=dict(l=20,r=20,t=50,b=20), hovermode="closest", autosize=True,
    ))
    return json.loads(fig.to_json())

