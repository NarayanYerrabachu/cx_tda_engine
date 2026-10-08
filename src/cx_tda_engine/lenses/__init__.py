"""Lens (filter) functions for Mapper: map X (n, d) to one value per row.

The lens decides what the Mapper cover slices along. Lens-independent stages
(homology, clustering, scoring) never see it, which is why switching lenses
is cheap in :func:`cx_tda_engine.run_full_pipeline`.
"""
from __future__ import annotations

from typing import Any

from .base import BaseLens
from .density import DensityLens
from .eccentricity import EccentricityLens
from .feature import FeatureLens
from .pca import PCALens
from .umap_lens import UMAPLens, umap_available

LENS_REGISTRY: dict[str, type[BaseLens]] = {
    "pca":          PCALens,
    "umap":         UMAPLens,
    "density":      DensityLens,
    "eccentricity": EccentricityLens,
    "feature":      FeatureLens,
}


def get_lens(name: str, **kwargs: Any) -> BaseLens:
    """Instantiate a registered lens by name."""
    cls = LENS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown lens '{name}'. Available: {list(LENS_REGISTRY)}")
    return cls(**kwargs)


def register_lens(name: str, cls: type[BaseLens]) -> None:
    """Add a custom lens to the registry (overrides an existing name)."""
    LENS_REGISTRY[name] = cls


def lens_catalog() -> list[dict[str, str]]:
    """Name and description of every registered lens, for UIs."""
    return [{"name": n, "description": c.description} for n, c in LENS_REGISTRY.items()]


__all__ = ["BaseLens", "PCALens", "UMAPLens", "DensityLens", "EccentricityLens", "FeatureLens",
           "LENS_REGISTRY", "get_lens", "register_lens", "lens_catalog", "umap_available"]
