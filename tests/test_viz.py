import numpy as np
import pytest

from cx_tda_engine import run_mapper

pytest.importorskip("plotly")


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    return run_mapper(X, X[:, 0], anomaly_scores=rng.random(300), record_ids=[str(i) for i in range(300)])


@pytest.mark.parametrize("layout", ["spring", "kamada_kawai", "shell", "spectral", "circular"])
def test_layout_json(graph, layout):
    from cx_tda_engine import viz

    out = viz.render_layout_json(graph["nodes"], graph["edges"], layout=layout)
    assert len(out["nodes"]) == len(graph["nodes"])
    assert all(-1.5 <= n["x"] <= 1.5 and -1.5 <= n["y"] <= 1.5 for n in out["nodes"])


def test_2d_3d_and_galaxy_figures(graph):
    from cx_tda_engine import viz

    fig2d = viz.render_plotly_json(graph["nodes"], graph["edges"])
    fig3d = viz.render_plotly_3d_json(graph["nodes"], graph["edges"])
    galaxy = viz.render_galaxy_3d_json(graph["nodes"], graph["edges"])
    for fig in (fig2d, fig3d, galaxy):
        assert fig["data"] and "layout" in fig
    assert len(galaxy["data"]) > len(fig3d["data"])          # gas layers on top of nodes + edges
    pos = viz.positions_3d(graph["nodes"], graph["edges"])
    assert len(pos) == len(graph["nodes"])


def test_png(graph):
    pytest.importorskip("matplotlib")
    from cx_tda_engine import viz

    png = viz.render_mapper_png(graph["nodes"], graph["edges"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
