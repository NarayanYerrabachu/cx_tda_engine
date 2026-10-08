from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import BaseLens


class PCALens(BaseLens):
    name = "pca"
    description = "Projects data onto the first principal component as filter function."

    def __init__(self, n_components: int = 1, seed: int = 42):
        self.n_components = n_components
        self._pca = PCA(n_components=n_components, random_state=seed)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        result = self._pca.fit_transform(X)
        return result[:, 0] if self.n_components == 1 else result
