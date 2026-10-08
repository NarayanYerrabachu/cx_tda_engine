"""Static Matplotlib PNG of a Mapper graph."""
from __future__ import annotations

import io

import networkx as nx
import numpy as np

from .graph import _build_nx
from .layouts import _layout


def render_mapper_png(nodes: list[dict], edges: list[dict], layout: str = "spring") -> bytes:
    """Static PNG of the Mapper graph (used by the Mapper PDF report when no
    interactive view is on screen to capture)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    G = _build_nx(nodes, edges)
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor="#13161E")
    ax.set_facecolor("#13161E"); ax.axis("off")
    if G.number_of_nodes():
        pos = _layout(G, layout)
        anom = np.array([G.nodes[n].get("avg_anomaly", 0.0) for n in G.nodes()])
        lo, hi = anom.min(), anom.max()
        cmap = LinearSegmentedColormap.from_list("mapper", ["#5E9CA6", "#9B7FD4", "#E05252"])
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color="#4A5060", width=0.8)
        coll = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[max(40, G.nodes[n].get("size", 1) ** 0.5 * 60) for n in G.nodes()],
                                      node_color=(anom - lo) / (hi - lo + 1e-9), cmap=cmap, vmin=0, vmax=1,
                                      alpha=0.9, linewidths=0.6, edgecolors="#1A1D24")
        coll.set_clip_on(False); ax.margins(0.08)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()

