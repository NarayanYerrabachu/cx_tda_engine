"""UMAP lens (optional extra ``cx_tda_engine[umap]``); falls back to PCA when umap-learn is missing."""
from __future__ import annotations

import importlib.util
import logging
import threading

import numpy as np

from .base import BaseLens

log = logging.getLogger(__name__)

# UMAP's compiled kernels (numba / pynndescent) are not safe to run in two
# threads at once: the process aborts. Every UMAP fit goes through ``embed``
# and takes this lock.
UMAP_LOCK = threading.Lock()


def umap_available() -> bool:
    return importlib.util.find_spec("umap") is not None


def embed(X: np.ndarray, n_components: int, n_neighbors: int = 15, min_dist: float = 0.1,
          seed: int = 42) -> np.ndarray:
    """UMAP embedding of X, one fit at a time per process. Raises ImportError without umap-learn."""
    import umap

    with UMAP_LOCK:
        return umap.UMAP(n_components=n_components, n_neighbors=min(n_neighbors, max(2, len(X) - 1)),
                         min_dist=min_dist, random_state=seed).fit_transform(X)


def warm_up() -> None:
    """Compile UMAP's kernels once (numba, several seconds) so the first real fit
    does not pay for it. Safe to call from a background thread at startup."""
    try:
        embed(np.random.default_rng(0).normal(size=(60, 4)), n_components=2, n_neighbors=5)
        log.info("UMAP ready")
    except ImportError:
        log.info("umap-learn not installed: the UMAP lens falls back to PCA")
    except Exception as exc:  # never more than a slower first request
        log.warning("UMAP warm-up failed: %s", exc)


class UMAPLens(BaseLens):
    name = "umap"
    description = "UMAP dimensionality reduction as filter function (non-linear)."

    def __init__(self, n_components: int = 1, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42):
        self.n_components = n_components
        self.n_neighbors  = n_neighbors
        self.min_dist     = min_dist
        self.seed         = seed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            result = embed(X, self.n_components, self.n_neighbors, self.min_dist, self.seed)
            return result[:, 0] if self.n_components == 1 else result
        except ImportError:
            log.warning("umap-learn not installed, falling back to PCA")
            from .pca import PCALens

            return PCALens(n_components=self.n_components, seed=self.seed).fit_transform(X)
