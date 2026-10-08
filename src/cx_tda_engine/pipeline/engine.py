"""``TDAEngine``: a facade that binds a config, a dataset spec and a cache together.

Functions are the primary API; the facade is for applications that run many
pipelines with the same settings (a service holding one engine per tenant).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import TDAConfig, resolve
from ..core.preprocessing import normalized_features
from ..core.types import PipelineResult
from ..lenses import get_lens
from ..mapper import run_mapper
from .cache import BaseCache, InMemoryBaseCache
from .runner import run_full_pipeline
from .spec import DatasetSpec


class TDAEngine:
    def __init__(self, config: TDAConfig | None = None, spec: DatasetSpec | None = None,
                 cache: BaseCache | None = None) -> None:
        self.config = resolve(config)
        self.spec = spec or DatasetSpec()
        self.cache: BaseCache = cache if cache is not None else InMemoryBaseCache()

    def run(self, df: pd.DataFrame, feature_cols: list[str], lens_name: str = "pca",
            n_intervals: int = 10, overlap: float = 0.5, *, spec: DatasetSpec | None = None,
            should_abort: Callable[[], bool] | None = None, **kwargs: Any) -> PipelineResult:
        """:func:`run_full_pipeline` with this engine's config, spec and cache."""
        return run_full_pipeline(df, feature_cols, lens_name, n_intervals, overlap,
                                 spec=spec or self.spec, config=self.config, cache=self.cache,
                                 should_abort=should_abort, **kwargs)

    def mapper(self, df: pd.DataFrame, feature_cols: list[str], lens_name: str = "pca",
               n_intervals: int = 10, overlap: float = 0.4, *, anomaly_scores: np.ndarray | None = None,
               record_ids: list[str] | None = None, lens_kwargs: dict[str, Any] | None = None,
               **mapper_kwargs: Any) -> dict[str, Any]:
        """Exploration Mapper graph straight from a DataFrame."""
        cols = [c for c in feature_cols if c in df.columns]
        X = normalized_features(df, cols)
        lens = np.asarray(get_lens(lens_name, **(lens_kwargs or {})).fit_transform(X), dtype=float)
        ids = record_ids if record_ids is not None else self.spec.record_ids(df)
        return run_mapper(X, lens, n_intervals, overlap, anomaly_scores=anomaly_scores, record_ids=ids,
                          config=self.config, **mapper_kwargs)

    def clear_cache(self) -> None:
        self.cache.clear()
