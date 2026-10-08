from __future__ import annotations

import numpy as np

from .base import BaseLens


class FeatureLens(BaseLens):
    name = "feature"
    description = "Uses a single (standardised) feature column as the filter function."

    def __init__(self, feature_index: int = 0):
        self.feature_index = feature_index

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        idx = min(self.feature_index, X.shape[1] - 1)
        return X[:, idx]
