"""Mapper (Singh, Memoli, Carlsson 2007): slice the lens image into overlapping
intervals, cluster each preimage, one node per local cluster, edges between
nodes that share records.

* :func:`build_mapper_graph`: the pipeline graph (stable, cheap, ``state`` per node).
* :func:`run_mapper`: the exploration graph (risk tiers, neighbours, histogram).
* :func:`dbscan_capped`: the memory guard both use.
"""
from .dbscan import AUTO_DEFAULT_EPS, auto_mapper_eps, dbscan_capped
from .explore import run_mapper
from .graph import build_mapper_graph

__all__ = ["build_mapper_graph", "run_mapper", "dbscan_capped", "auto_mapper_eps", "AUTO_DEFAULT_EPS"]
