"""Cache of lens-independent results, so switching the lens is a Mapper rebuild only.

:class:`BaseCache` is the protocol; :class:`InMemoryBaseCache` the default
implementation. An application can provide its own (e.g. persisted to disk
and restored after a restart) as long as it honours the protocol.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

CacheKey = tuple[Any, ...]


def cache_key(df: pd.DataFrame, feature_cols: list[str]) -> CacheKey:
    """(sorted feature columns, row count).

    Cell values are deliberately not hashed: hashing a large frame on every
    call would cost more than the cache saves. Callers clear the cache when
    the dataset changes.
    """
    return (tuple(sorted(c for c in feature_cols if c in df.columns)), len(df))


@runtime_checkable
class BaseCache(Protocol):
    def get(self, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any] | None: ...
    def seed(self, df: pd.DataFrame, feature_cols: list[str], base: dict[str, Any]) -> None: ...
    def clear(self) -> None: ...


class InMemoryBaseCache:
    """Process-local dict cache."""

    def __init__(self) -> None:
        self._store: dict[CacheKey, dict[str, Any]] = {}

    @staticmethod
    def key(df: pd.DataFrame, feature_cols: list[str]) -> CacheKey:
        return cache_key(df, feature_cols)

    def get(self, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any] | None:
        return self._store.get(cache_key(df, feature_cols))

    def seed(self, df: pd.DataFrame, feature_cols: list[str], base: dict[str, Any]) -> None:
        self._store[cache_key(df, feature_cols)] = base

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


#: Process-wide default used when ``run_full_pipeline`` gets no ``cache``.
DEFAULT_CACHE = InMemoryBaseCache()


def clear_base_cache() -> None:
    """Drop the default cache. Call when the underlying dataset changes."""
    DEFAULT_CACHE.clear()


def cached_base(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any] | None:
    """The default cache's lens-independent results for this dataset, if computed."""
    return DEFAULT_CACHE.get(df, feature_cols)


def seed_base_cache(df: pd.DataFrame, feature_cols: list[str], base: dict[str, Any]) -> None:
    """Put restored lens-independent results into the default cache."""
    DEFAULT_CACHE.seed(df, feature_cols, base)
