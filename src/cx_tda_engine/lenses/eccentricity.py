from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from .base import BaseLens


class EccentricityLens(BaseLens):
    name = "eccentricity"
    description = "Eccentricity: mean distance from each point to (a sample of) all others."

    def __init__(self, metric: str = "euclidean", sample_n: int = 1000, seed: int = 42):
        self.metric   = metric
        self.sample_n = sample_n
        self.seed     = seed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n > self.sample_n:
            idx = np.random.default_rng(self.seed).choice(n, self.sample_n, replace=False)
            X_ref = X[idx]
        else:
            X_ref = X
        return pairwise_distances(X, X_ref, metric=self.metric).mean(axis=1)
