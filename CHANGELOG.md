# Changelog

## 0.1.0 (2026-10-08)

Initial extraction from the CortXplorer TDA demo.

* `run_full_pipeline` with the demo's result contract, lens-independent base cache and abortable stages.
* Lenses: pca, umap (optional), density, eccentricity, feature; registry with `register_lens`.
* `build_mapper_graph` (pipeline) and `run_mapper` (exploration) with memory-capped DBSCAN.
* Persistent homology via ripser with noise-scale Betti numbers and loop classes.
* Isolation Forest + topological anomaly scores, DBSCAN clustering with auto eps, relationships, drift, suspicious rules.
* `TDAConfig` (replaces `TDA_*` env vars and schema constants) and `DatasetSpec` (replaces hard-coded column names).
* `text` module: `Document`, chunking, TF-IDF / SVD vectorisation, `run_text_pipeline`, processor registry.
* Layered package layout (core / analysis / mapper / pipeline / adapters / viz), stage chain, cache protocol, `TDAEngine` facade, `make_finding`.
* PyPI-ready packaging: MIT license, classifiers, `py.typed`, manual trusted-publishing workflow.
* Optional `viz` extra: 2-D / 3-D / galaxy Plotly figures, Matplotlib PNG, layout JSON.
