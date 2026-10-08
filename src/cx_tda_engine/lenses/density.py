from __future__ import annotations

import numpy as np
from sklearn.neighbors import KernelDensity

from .base import BaseLens


class DensityLens(BaseLens):
    name = "density"
    description = "Local kernel density estimate: highlights dense vs. sparse regions."

    def __init__(self, bandwidth: float = 0.5):
        self.bandwidth = bandwidth

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian").fit(X)
        return kde.score_samples(X)  # log density, shape (n_samples,)
