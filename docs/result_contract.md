# Result contract of `run_full_pipeline`

The return value is a plain dict (JSON-serialisable). `cx_tda_engine.types`
describes it as TypedDicts. On a bad input the dict has a single `error` key.

## Top level

| key | type | content |
|---|---|---|
| `meta` | dict | counts, thresholds, effective parameters, `tda` block (see below) |
| `themes` | list[Finding] | one per DBSCAN cluster, `kind="theme"` |
| `anomalies` | list[Finding] | top `anomaly_top_k` records by combined score, `kind="anomaly"` |
| `suspicious` | list[Finding] | top `suspicious_top_k` rule-flagged records, `kind="suspicious"` |
| `relationships` | list[Finding] | up to 60 label co-occurrences, `kind="relationship"` |
| `drift` | list[Finding] | feature or metric shifts, `kind="drift"` |
| `topology` | list[Finding] | H1 loops above the noise band, `kind="topology"` |
| `graph.nodes` / `graph.edges` | list | relationship graph (`{id, docs, exposure}` / `{a, b, w, sources}`) |
| `graph.mapper_nodes` / `graph.mapper_edges` | list | pipeline Mapper graph (below) |
| `documents` | list | always empty (kept for the frontend contract) |
| `record_ids` | list[str] | one per row, from `DatasetSpec.id_col` or the row position |
| `lens_values` | list[float] | lens value per row |
| `cluster_labels` | list[int] | DBSCAN label per row, -1 = noise |
| `combined_scores` | list[float] | anomaly score per row in [0, 1] |

## Finding

```
{"kind": str, "title": str, "score": float, "sources": [record_id, ...], "detail": str, "extra": {...}}
```

`extra` per kind:

* anomaly: `iso_score`, `ae_score` (always null), `topo_score`, `flagged_by` (3 feature names), `priority` (HIGH/MEDIUM/REVIEW), plus `display_cols` and the first six feature values.
* suspicious: `reasons` (list[str]), `iso_score`, `topo_score`, `priority` (HIGH/MEDIUM), plus `display_cols`.
* theme: `cluster_id`, `n_records`, `stats` (`{col: {mean, std}}` for `stat_cols`).
* relationship: `a`, `b`, `weight`, `type`, `n_anomalous`, `n_suspicious`, `a_stats`, `b_stats` (`{n_records, n_links, n_anomalous, n_suspicious}`).
* drift: `feature`, `early_mean`, `late_mean`, `delta`, `mid_year` (null for cluster drift), `direction`.
* topology: the H1 bar (`birth`, `death`, `persistence`, `label`, `significant`).

## Mapper node (`graph.mapper_nodes`)

`id` (`"node-<i>"`), `size`, `sources` (first 20 record ids), `interval`, `cluster`, `lens_mean`,
and when scores are available: `avg_anomaly`, `max_anomaly`, `frac_anomalous`, `n_anomalous`,
`node_score`, `lift` (null when the dataset has no anomalous record), `state`
(`anomalous` / `warning` / `normal`). Edges: `{source, target, weight}` with node indices.

## `meta`

`n_docs`, `n_clusters`, `n_noise_docs`, `n_anomalies` (shown), `n_anomalies_total`,
`n_suspicious` (shown), `n_suspicious_total`, `anomaly_threshold`, `n_relationships`,
`n_drift`, `n_topology`, `n_themes`, `graph_backend` (`"in-memory"`), `ripser_available`,
`theme_quality` (null), `numeric_features`, `category_features`, `data_type` (`"tabular"`),
`lens`, `cached_base`, `elapsed_s`, `mapper_n_intervals`, `mapper_overlap`, `mapper_requested`,
`mapper_min_n_intervals`, `mapper_min_overlap`, `seed`, `contamination`, `engine_version`.

### `meta.tda` (persistent homology)

`available`, `betti_0`, `betti_1` (at the noise scale), `betti_0_raw`, `betti_1_raw`,
`noise_threshold`, `median_h1_persistence`, `max_persistence`, `loop_class`
(`acyclic` / `weak` / `moderate` / `high`), `persistence_ratio`, `h0_features`,
`h1_features` (labelled significant loops), `diagram_h0`, `diagram_h1`, `diagram_max`.

## `run_mapper` (exploration graph)

`nodes` (`id`, `size`, `lens_center`, `lens_low`, `lens_high`, `interval_idx`, `interval_lo`,
`interval_hi`, `avg_anomaly`, `max_anomaly`, `n_anomalous`, `frac_anomalous`, `lift`,
`risk_level` HIGH/WATCH/LOW, `members`, `n_members`, `neighbors`), `edges`, `stats`
(`n_nodes`, `n_edges`, `n_components`, `n_intervals`, `overlap`, `dbscan_eps`, sizes,
`l_min`, `l_max`, `anomaly_threshold`, `anomaly_base_rate`, `risk_high_lift`,
`risk_watch_lift`, `n_high`, `n_watch`, `n_low`, `records_*`), `lens_histogram`, `interval_bands`.
