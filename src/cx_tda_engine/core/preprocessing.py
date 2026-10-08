"""Standardisation of the feature matrix every stage runs on."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance columns; NaN and inf become 0 first."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(X)


def normalized_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """The standardised feature matrix (``X_norm``) for the given columns."""
    return normalize(df[feature_cols].to_numpy(dtype=float))
