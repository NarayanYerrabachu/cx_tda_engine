"""From a DataFrame to the result contract.

* :mod:`spec`: what the table's columns mean (:class:`DatasetSpec`).
* :mod:`stages`: the lens-independent work as a chain of stages.
* :mod:`cache`: where the stage results are kept between lens switches.
* :mod:`runner`: :func:`run_full_pipeline`, the functional entry point.
* :mod:`engine`: :class:`TDAEngine`, a facade binding config, spec and cache.
"""
from .cache import (
    DEFAULT_CACHE,
    BaseCache,
    InMemoryBaseCache,
    cache_key,
    cached_base,
    clear_base_cache,
    seed_base_cache,
)
from .engine import TDAEngine
from .runner import run_full_pipeline
from .spec import DatasetSpec
from .stages import DEFAULT_STAGES, BaseContext, PipelineSuperseded, Stage, compute_base, run_stages

__all__ = [
    "DatasetSpec", "run_full_pipeline", "TDAEngine",
    "BaseCache", "InMemoryBaseCache", "DEFAULT_CACHE", "cache_key", "cached_base", "clear_base_cache", "seed_base_cache",
    "BaseContext", "Stage", "DEFAULT_STAGES", "run_stages", "compute_base", "PipelineSuperseded",
]
