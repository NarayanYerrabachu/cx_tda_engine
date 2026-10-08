"""Drift: how feature distributions shift over time or between the dominant clusters."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finding(title: str, detail: str, feature: str, early: float, late: float,
             mid: int | None, direction: str) -> dict[str, Any]:
    delta = late - early
    return {
        "kind": "drift", "title": title, "score": round(abs(delta), 4), "sources": [], "detail": detail,
        "extra": {"feature": feature, "early_mean": round(early, 4), "late_mean": round(late, 4),
                  "delta": round(delta, 4), "mid_year": mid, "direction": direction},
    }


def cluster_drift(X_norm: np.ndarray, feature_cols: list[str], cluster_labels: np.ndarray,
                  top_k: int = 8, min_delta: float = 0.15) -> list[dict[str, Any]]:
    """Distributional drift without a time column: mean of each standardised
    feature in the second-largest cluster vs. the largest. Deltas are in
    standard-deviation units; the non-temporal analogue of early-vs-late drift."""
    findings: list[dict[str, Any]] = []
    if not feature_cols or X_norm.shape[0] < 20 or X_norm.shape[1] < 1:
        return findings
    labels = np.asarray(cluster_labels)
    sizes = sorted(((int((labels == c).sum()), int(c)) for c in set(labels.tolist()) if c != -1), reverse=True)
    if len(sizes) < 2:
        return findings
    (n_a, cid_a), (n_b, cid_b) = sizes[0], sizes[1]
    a_rows, b_rows = X_norm[labels == cid_a], X_norm[labels == cid_b]

    for j, col in enumerate(feature_cols[:X_norm.shape[1]]):
        a_mean, b_mean = float(a_rows[:, j].mean()), float(b_rows[:, j].mean())
        delta = b_mean - a_mean
        if abs(delta) < min_delta:
            continue
        higher = "higher" if delta > 0 else "lower"
        name = col.replace("_", " ")
        findings.append(_finding(
            f"{name}: cluster #{cid_b} {higher} than cluster #{cid_a}",
            f"{name} is {abs(delta):.2f}σ {higher} in cluster #{cid_b} ({n_b} records, mean {b_mean:.2f}) "
            f"vs cluster #{cid_a} ({n_a} records, mean {a_mean:.2f}).",
            col, a_mean, b_mean, None, "increased" if delta > 0 else "decreased"))
    findings.sort(key=lambda x: abs(x["extra"]["delta"]), reverse=True)
    return findings[:top_k]


def temporal_feature_drift(df: pd.DataFrame, X_norm: np.ndarray, feature_cols: list[str], time_col: str,
                           top_k: int = 8, min_delta: float = 0.1) -> list[dict[str, Any]]:
    """Split records at the median of ``time_col`` (numeric, e.g. a year) and
    compare each standardised feature's mean between the early and late halves."""
    findings: list[dict[str, Any]] = []
    years = pd.to_numeric(df[time_col], errors="coerce")
    valid = years.notna().to_numpy()
    if valid.sum() < 20 or not feature_cols:
        return findings
    mid = float(years.dropna().median())
    early = (years <= mid).to_numpy() & valid
    late = (years > mid).to_numpy() & valid
    if early.sum() < 5 or late.sum() < 5:
        return findings

    for j, col in enumerate(feature_cols[:X_norm.shape[1]]):
        e_mean, l_mean = float(X_norm[early, j].mean()), float(X_norm[late, j].mean())
        delta = l_mean - e_mean
        if abs(delta) < min_delta:
            continue
        direction = "increased" if delta > 0 else "decreased"
        name = col.replace("_", " ")
        findings.append(_finding(
            f"{name} drift: {direction} over time",
            f"{name} {direction} by {abs(delta):.2f}σ from the pre-{int(mid)} period (mean {e_mean:.2f}) "
            f"to the post-{int(mid)} period (mean {l_mean:.2f}), standardised.",
            col, e_mean, l_mean, int(mid), direction))
    findings.sort(key=lambda x: abs(x["extra"]["delta"]), reverse=True)
    return findings[:top_k]


def temporal_metric_drift(df: pd.DataFrame, time_col: str, metric_cols: dict[str, str],
                          min_rows: int = 10, min_delta: float = 0.001) -> list[dict[str, Any]]:
    """Early-vs-late shift of raw (unstandardised) domain metrics, e.g.
    ``{"cost_overrun_pct": "Cost Overrun %"}``. Split at the median of ``time_col``."""
    findings: list[dict[str, Any]] = []
    years = pd.to_numeric(df[time_col], errors="coerce")
    if years.notna().sum() < min_rows:
        return findings
    mid = float(years.dropna().median())
    early, late = df[years <= mid], df[years > mid]
    for col, label in metric_cols.items():
        if col not in df.columns:
            continue
        e = pd.to_numeric(early[col], errors="coerce").dropna()
        lt = pd.to_numeric(late[col], errors="coerce").dropna()
        e_mean = float(e.mean()) if len(e) else 0.0
        l_mean = float(lt.mean()) if len(lt) else 0.0
        delta = l_mean - e_mean
        if abs(delta) < min_delta:
            continue
        direction = "increased" if delta > 0 else "decreased"
        findings.append(_finding(
            f"{label} drift: {direction} over time",
            f"{label} {direction} from {e_mean:.3f} (pre-{int(mid)}) to {l_mean:.3f} (post-{int(mid)}).",
            col, e_mean, l_mean, int(mid), direction))
    findings.sort(key=lambda x: abs(x["extra"]["delta"]), reverse=True)
    return findings


def compute_drift(
    df: pd.DataFrame,
    X_norm: np.ndarray,
    cluster_labels: np.ndarray,
    feature_cols: list[str] | None = None,
    time_col: str | None = None,
    metric_cols: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Drift findings for a dataset.

    Order of preference: raw-metric drift when ``metric_cols`` and ``time_col``
    are given; otherwise standardised-feature drift over ``time_col``; and when
    neither yields anything, drift between the two largest clusters, so the
    lens is never empty on a dataset with structure.
    """
    feature_cols = feature_cols or []
    has_time = time_col is not None and time_col in df.columns
    if has_time and metric_cols:
        found = temporal_metric_drift(df, str(time_col), metric_cols)
        if found:
            return found
    if has_time and feature_cols:
        found = temporal_feature_drift(df, X_norm, feature_cols, str(time_col))
        if found:
            return found
    return cluster_drift(X_norm, feature_cols, cluster_labels)
