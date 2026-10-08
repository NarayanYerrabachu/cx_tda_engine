"""Abstract base class for all lenses."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseLens(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Map X (n_samples, n_features) to filter values of shape (n_samples,) or (n_samples, k)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
