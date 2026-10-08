"""Suspicious records: rule-based reasons on top of the anomaly score."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import TDAConfig, resolve
from ..core.types import SuspiciousRule

_MISSING = ("nan", "None", "Unknown", "N/A", "")


def score_rule(config: TDAConfig | None = None) -> SuspiciousRule:
    """The default rule: flag records whose combined score is at or above ``anomaly_high``."""
    cfg = resolve(config)

    def _rule(row: pd.Series, score: float) -> list[str]:
        return [f"High composite anomaly score ({score:.2f})"] if score >= cfg.anomaly_high else []

    return _rule


def build_suspicious_findings(
    df: pd.DataFrame,
    combined_scores: np.ndarray,
    record_ids: list[str],
    top_k: int | None = 20,
    iso_scores: np.ndarray | None = None,
    topo_scores: np.ndarray | None = None,
    rules: list[SuspiciousRule] | None = None,
    group_cols: list[str] | None = None,
    display_cols: list[str] | None = None,
    config: TDAConfig | None = None,
) -> list[dict[str, Any]]:
    """``kind="suspicious"`` findings for every record at least one rule flags.

    ``rules`` default to :func:`score_rule`; a caller with domain knowledge
    passes its own (each returns the reasons for one row given its score) and
    the default is then NOT applied. ``top_k=None`` returns every match so the
    caller can report the true total; the list is sorted by score.
    """
    cfg = resolve(config)
    rules = rules if rules is not None else [score_rule(cfg)]
    group_cols = [c for c in (group_cols or []) if c in df.columns]
    display_cols = [c for c in (display_cols or []) if c in df.columns]

    suspicious: list[dict[str, Any]] = []
    for i, pid in enumerate(record_ids):
        if i >= len(df):
            break
        row = df.iloc[i]
        score = float(combined_scores[i]) if i < len(combined_scores) else 0.0
        reasons: list[str] = []
        for rule in rules:
            reasons.extend(rule(row, score))
        if not reasons:
            continue

        gparts = [str(row.get(c)) for c in group_cols[:2]
                  if row.get(c) is not None and str(row.get(c)).strip() not in _MISSING]
        title = f"{pid} — " + ", ".join(gparts) if gparts else str(pid)
        iso_v = float(iso_scores[i]) if iso_scores is not None and i < len(iso_scores) else None
        topo_v = float(topo_scores[i]) if topo_scores is not None and i < len(topo_scores) else None
        extra: dict[str, Any] = {"reasons": reasons}
        for c in display_cols:
            v = row.get(c)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                extra[c] = round(float(v), 4) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
        extra.update({
            "iso_score":  round(iso_v, 4) if iso_v is not None else None,
            "topo_score": round(topo_v, 4) if topo_v is not None else None,
            "priority":   "HIGH" if score >= cfg.anomaly_high else "MEDIUM",
        })
        suspicious.append({"kind": "suspicious", "title": title, "score": round(score, 4),
                           "sources": [pid], "detail": "; ".join(reasons), "extra": extra})

    suspicious.sort(key=lambda x: x["score"], reverse=True)
    return suspicious if top_k is None else suspicious[:top_k]
