"""Mapper graph renderers (optional extra ``cx_tda_engine[viz]``).

Input is the ``nodes`` / ``edges`` of :func:`cx_tda_engine.run_mapper`. Plotly
and Matplotlib are imported lazily inside the render functions, so the rest
of the library never needs them.
"""
from .layouts import positions_3d, render_layout_json
from .plotly2d import render_plotly_json
from .plotly3d import render_galaxy_3d_json, render_plotly_3d_json
from .png import render_mapper_png

__all__ = ["render_plotly_json", "render_plotly_3d_json", "render_galaxy_3d_json", "render_mapper_png",
           "render_layout_json", "positions_3d"]
