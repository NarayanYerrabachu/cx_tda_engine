"""What the engine needs to know about a table beyond its numeric features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..core.types import SuspiciousRule


@dataclass(frozen=True)
class DatasetSpec:
    """Column meaning and domain hooks for one kind of dataset.

    Everything is optional: with the defaults the pipeline is purely numeric
    and titles records by their row position. Applications define one spec
    per dataset family (see ``docs/migration_from_demo.md`` for the demo's).
    """

    #: Column holding the record identifier (string-cast); None = row position.
    id_col: str | None = None
    #: Up to two categorical columns used to title clusters and anomalies and
    #: to build the relationship graph (``a`` x ``b`` co-occurrence).
    group_cols: tuple[str, ...] = ()
    #: Numeric column (e.g. a year) that splits early vs. late for drift.
    time_col: str | None = None
    #: Columns copied into ``extra`` of anomaly and suspicious findings for display.
    display_cols: tuple[str, ...] = ()
    #: Columns summarised (mean/std) per cluster theme.
    stat_cols: tuple[str, ...] = ()
    #: Raw metric columns for drift, ``{column: label}``; needs ``time_col``.
    metric_cols: dict[str, str] = field(default_factory=dict)
    #: Rules deciding which records are suspicious (default: score >= anomaly_high).
    suspicious_rules: tuple[SuspiciousRule, ...] | None = None
    #: Rows the relationship co-occurrence is counted over (default: all rows).
    relationship_basis: Callable[[pd.DataFrame], np.ndarray] | None = None
    relationship_type: str = "cooccurrence"

    def record_ids(self, df: pd.DataFrame) -> list[str]:
        if self.id_col and self.id_col in df.columns:
            return df[self.id_col].astype(str).tolist()
        return [str(i) for i in range(len(df))]

    def present_group_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.group_cols if c in df.columns]
