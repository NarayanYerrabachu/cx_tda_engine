# Migrating the CortXplorer demo to `cx_tda_engine`

The demo's `backend/tda/` package maps onto the library as follows.

| demo | library |
|---|---|
| `backend.tda.engine._normalize` / `normalized_features` | `cx_tda_engine.normalize` / `normalized_features` |
| `engine.homology_projection`, `compute_persistent_homology` | `cx_tda_engine.homology` (same names) |
| `engine.build_mapper_graph` | `cx_tda_engine.build_mapper_graph` (drops the unused `cluster_labels` arg) |
| `engine.detect_anomalies(..., cat_cols, id_col)` | `detect_anomalies(..., group_cols, display_cols)` |
| `engine._auto_eps`, `compute_clusters(..., cat_cols, id_col)` | `auto_eps`, `compute_clusters(..., group_cols, stat_cols)` |
| `engine.compute_relationships` | `compute_relationships(..., group_cols, basis_mask, relationship_type)` |
| `engine.compute_drift` | `compute_drift(..., time_col, metric_cols)` |
| `engine.build_suspicious_findings` | `build_suspicious_findings(..., rules, group_cols, display_cols)` |
| `engine.run_full_pipeline(df, cols, lens, n, ov, use_cache, should_abort)` | same positional args, plus `spec=`, `config=`, `cache=` |
| `engine.cached_base` / `seed_base_cache` / `clear_base_cache` / `PipelineSuperseded` | same names (`cx_tda_engine.pipeline`) |
| `backend.tda.mapper.run_mapper` | `cx_tda_engine.run_mapper` (+ `stats.dbscan_eps`) |
| `backend.tda.mapper_viz` | `cx_tda_engine.viz` (extra `viz`) |
| `backend.tda.lenses` | `cx_tda_engine.lenses` (+ `register_lens`, `lens_catalog`, `umap_available`) |
| `gov_aid_schema.ANOMALY_HIGH_THRESHOLD` etc. | `TDAConfig.anomaly_high`, `high_lift`, `watch_lift`, `score_frac_weight`, `loop_*_max` |
| `gov_aid_schema.graph_node_state`, `mapper_node_risk_level`, `mapper_risk_from_lift`, `node_risk_level`, `mapper_risk_counts`, `loop_class` | `cx_tda_engine.risk` (same names, optional `config=`) |
| `TDA_*` environment variables | `TDAConfig.from_env()` |
| `backend.text_pipeline.chunk_documents` / `documents_to_dataframe` | `cx_tda_engine.text` (same names; `record_id` instead of `Project_ID`; optional SVD) |
| `backend.processors.base.Document` / `DocumentProcessor`, registry | `cx_tda_engine.text` (same names); the PDF/Word/email/ZIP processors stay in the app |

## The gov-aid `DatasetSpec`

```python
from cx_tda_engine import DatasetSpec
from backend.data import gov_aid_schema as s

def gov_aid_rules(row, score):
    reasons = []
    ovr = row.get(s.COST_OVERRUN_PCT)
    if isinstance(ovr, float) and ovr == ovr:
        if ovr > s.OVERRUN_EXTREME_THRESHOLD:
            reasons.append(f"Extreme overrun ({ovr*100:.1f}%)")
        elif ovr > 1.0 and score > 0.5:
            reasons.append(f"High overrun ({ovr*100:.1f}%)")
    if row.get(s.SUCCESS) == 0 and score > 0.5:
        reasons.append("Failed project")
    cpi = row.get(s.CPI_SCORE)
    if isinstance(cpi, float) and cpi == cpi and cpi < 250:
        reasons.append(f"Low CPI ({cpi:.0f})")
    if "unusual" in str(row.get(s.BUDGET_UNUSUAL, "")).lower():
        reasons.append("Unusual budget")
    return reasons

GOV_AID_SPEC = DatasetSpec(
    id_col=s.ID_COL,
    group_cols=(s.COUNTRY_COL, s.DAC_MAPPING),
    time_col=s.APPROVAL_YEAR,
    display_cols=(s.COUNTRY_COL, s.DAC_MAPPING, s.COST_OVERRUN_PCT, s.SUCCESS, s.CPI_SCORE,
                  s.EVAL_LAG, s.BUDGET_INIT, s.APPROVAL_YEAR),
    stat_cols=(s.COST_OVERRUN_PCT, s.CPI_SCORE, s.EVAL_LAG, s.SUCCESS),
    metric_cols={s.COST_OVERRUN_PCT: "Cost Overrun %", s.CPI_SCORE: "CPI Score", s.SUCCESS: "Project Success Rate"},
    suspicious_rules=(gov_aid_rules,),
    relationship_basis=lambda df: (df[s.COST_OVERRUN_PCT] > 0.5).to_numpy(),
    relationship_type="overrun_cooccurrence",
)
```

Generic datasets (financial, uploads) use `DatasetSpec(id_col=..., group_cols=(a, b), time_col=year)`
with no rules: that reproduces the demo's "generic" branches (score-only suspicion, all-rows
co-occurrence, year-feature drift with cluster-drift fallback).

## Behaviour changes to be aware of

* **Anomaly `extra` aliases are gone.** The demo wrote `country`, `dac_sector`,
  `cost_overrun_pct`, `success`, `cpi_score`, `eval_lag` into every anomaly and
  suspicious finding. The library writes the `display_cols` under their own
  column names. Add the aliases in the demo's adapter layer (one dict
  comprehension over `result["anomalies"]`) or update the frontend keys.
* **Theme labels** no longer append "avg overrun 12%" / "avg CPI 300"; the
  numbers are in `extra.stats` for the `stat_cols` instead. Cluster `detail`
  always says "records", never "aid projects".
* **Cluster membership in the relationship graph** uses the first `group_col`
  (the demo hard-coded the country column); identical for gov-aid.
* **Drift precedence**: with `metric_cols` + `time_col` the raw-metric drift is
  used when it yields findings, otherwise standardised-feature drift, otherwise
  cluster drift. The demo returned an empty list for gov-aid data without a
  year column; the library falls through to cluster drift.
* **`build_mapper_graph`** lost its unused `cluster_labels` parameter.
* **Eccentricity lens** sampling (n > 1000) is now seeded (reproducible).
* **`run_mapper` stats** gained `dbscan_eps`; `compute_persistent_homology`
  takes `config=`; all `TDA_*` env vars are read only through `TDAConfig.from_env()`.
* **Text tables** name the id column `record_id`, not `Project_ID`; pass `spec=TEXT_SPEC`
  (or your own `DatasetSpec`) to `run_full_pipeline`, or use `run_text_pipeline`.
* `meta.torch_available`, `giotto_available`, `neo4j_available` were dropped (always False).
  `meta.engine_version` was added.

## Numerical equivalence

For the same DataFrame, feature list, lens and seed, the library produces the
same `combined_scores`, `cluster_labels` and Mapper node membership as the
demo's engine: the algorithms, seeds, PCA dimensions, eps heuristics and fit
caps were ported unchanged. Regenerate a fixture from the demo venv to assert
this in the demo's own test-suite when it adopts the library.
