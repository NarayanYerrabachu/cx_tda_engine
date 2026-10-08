# cx_tda_engine

Topological data analysis for tabular data: Mapper graphs, persistent homology,
lens functions, anomaly scores, clusters, relationships and drift, returned as
one JSON-serialisable result. Extracted from the CortXplorer TDA demo and freed
of any dataset-specific assumptions.

## Install

From GitHub (any machine with git access to the repo):

```bash
pip install "cx_tda_engine @ git+https://github.com/NarayanYerrabachu/cx_tda_engine.git"
pip install "cx_tda_engine[viz] @ git+https://github.com/NarayanYerrabachu/cx_tda_engine.git"   # + Plotly/Matplotlib renderers
pip install "cx_tda_engine[umap] @ git+https://github.com/NarayanYerrabachu/cx_tda_engine.git"  # + UMAP lens
pip install "cx_tda_engine @ git+https://github.com/NarayanYerrabachu/cx_tda_engine.git@v0.1.0" # pin a release tag
```

In a `requirements.txt` or `Pipfile`:

```
cx_tda_engine @ git+https://github.com/NarayanYerrabachu/cx_tda_engine.git@v0.1.0
```
```toml
cx_tda_engine = {git = "https://github.com/NarayanYerrabachu/cx_tda_engine.git", ref = "v0.1.0", extras = ["viz"]}
```

From a release wheel: every `v*` tag builds a wheel and attaches it to the
GitHub release, so `pip install https://github.com/NarayanYerrabachu/cx_tda_engine/releases/download/v0.1.0/cx_tda_engine-0.1.0-py3-none-any.whl`
works without git. Publishing to PyPI or a private index is one `twine upload dist/*` away
(`python -m build` produces `dist/`).

Development (pipenv, virtualenv named `cx_tda_engine`):

```bash
export PIPENV_CUSTOM_VENV_NAME=cx_tda_engine
pipenv install --dev
pipenv run pytest -q
pipenv run ruff check . && pipenv run mypy
```

## Quickstart

```python
import pandas as pd
from cx_tda_engine import DatasetSpec, TDAConfig, run_full_pipeline

df = pd.read_csv("records.csv")
spec = DatasetSpec(
    id_col="record_id",                 # how records are named in findings
    group_cols=("region", "category"),  # label clusters/anomalies, build relationships
    time_col="year",                    # early-vs-late drift
    display_cols=("region", "amount"),  # copied into finding.extra for display
)
result = run_full_pipeline(
    df, ["amount", "duration", "score"],
    lens_name="pca", n_intervals=15, overlap=0.5,
    spec=spec, config=TDAConfig(anomaly_top_k=50),
)

result["meta"]["n_anomalies_total"]     # records with combined score >= 0.60
result["anomalies"][0]["title"]         # "R0017 — south, C"
result["graph"]["mapper_nodes"]         # Mapper nodes with state / lift / node_score
result["meta"]["tda"]["betti_1"]        # loops above the noise band
```

Switching the lens reuses the cached lens-independent work, so the second
call below is a sub-second Mapper rebuild:

```python
run_full_pipeline(df, cols, lens_name="pca", spec=spec)
run_full_pipeline(df, cols, lens_name="eccentricity", spec=spec)   # meta.cached_base == True
```

## Exploration Mapper and renderers

```python
from cx_tda_engine import get_lens, normalized_features, run_mapper
from cx_tda_engine import viz   # needs cx_tda_engine[viz]

X = normalized_features(df, cols)
lens = get_lens("density", bandwidth=0.5).fit_transform(X)
graph = run_mapper(X, lens, n_intervals=12, overlap=0.4,
                   anomaly_scores=result["combined_scores"], record_ids=result["record_ids"])

graph["stats"]              # n_nodes, n_components, risk tiers, lens range
graph["lens_histogram"]     # for the filter panel
viz.render_plotly_json(graph["nodes"], graph["edges"], layout="spring")   # 2-D figure dict
viz.render_plotly_3d_json(graph["nodes"], graph["edges"])                 # 3-D figure dict
viz.render_galaxy_3d_json(graph["nodes"], graph["edges"])                 # 3-D "galaxy" with gas layers
viz.render_mapper_png(graph["nodes"], graph["edges"])                     # PNG bytes (Matplotlib)
viz.render_layout_json(graph["nodes"], graph["edges"], layout="kamada_kawai")  # positions only
```

## Unstructured data (documents, free text)

```python
from cx_tda_engine import Document, run_text_pipeline

docs = [Document(source_file="q3.pdf", file_name="q3.pdf", file_type="pdf", content=text_of_q3), ...]
result, df, feature_cols = run_text_pipeline(docs, lens_name="pca", n_components=20)
result, df, feature_cols = run_text_pipeline(["plain string one", "plain string two", ...])
```

Every document (or chunk, when a corpus has fewer than `min_rows` documents)
becomes a row: TF-IDF over the text, compressed with truncated SVD to
`n_components` latent dimensions (`lsa_1..k`), plus `file_name`, `file_type`,
`word_count` and `parent_file` as display and grouping columns. The result is
the same contract as the tabular pipeline. `documents_to_dataframe` and
`chunk_documents` are available separately; `n_components=None` keeps the raw
`tfidf_<term>` columns.

File parsing stays with the application: subclass `DocumentProcessor` for
PDF, Word, e-mail or ZIP and `register_processor()` it; `detect_processor(ext)`
dispatches by extension.

## Lenses

| name | what it measures | notes |
|---|---|---|
| `pca` | first principal component | default; linear, fast |
| `umap` | 1-D UMAP embedding | extra `umap`; falls back to PCA when absent; fits are serialised with a lock |
| `density` | Gaussian kernel log-density | dense cores vs. sparse fringes |
| `eccentricity` | mean distance to (a sample of) all points | centre vs. periphery |
| `feature` | one standardised column | `get_lens("feature", feature_index=2)` |

Add your own with `register_lens("name", MyLens)`; a lens is any class with
`fit_transform(X) -> (n,)` array.

## Configuration

`TDAConfig` holds every knob (seed, ripser sample size, Isolation Forest
contamination, DBSCAN eps / min_samples, worker count, Mapper fit caps,
display caps, anomaly thresholds, lift tiers, loop-class bounds, Mapper
resolution floors). `TDAConfig.from_env()` reads the `TDA_*` environment
variables the demo already uses. Pass `config=` to any function; the module
default is used otherwise.

## Domain hooks

The engine never assumes column names. Domain behaviour is injected through `DatasetSpec`:

* `suspicious_rules`: callables `(row, score) -> [reasons]`; the default flags score >= `anomaly_high`.
* `relationship_basis`: `df -> bool mask` selecting the rows co-occurrence is counted over.
* `metric_cols`: `{column: label}` raw metrics compared early vs. late over `time_col`.
* `stat_cols`: per-cluster mean/std summaries.

See `docs/result_contract.md` for the output shape and
`docs/migration_from_demo.md` for how the CortXplorer demo maps onto this API.

## Package layout

```
cx_tda_engine/
  config.py      TDAConfig: every numeric knob, TDAConfig.from_env()
  core/          result types + make_finding, risk rules (lifts, loop classes), preprocessing
  lenses/        Strategy + registry: BaseLens, get_lens, register_lens
  analysis/      lens-independent stages: homology, clustering, anomalies, relationships, drift, suspicious
  mapper/        build_mapper_graph (pipeline), run_mapper (exploration), dbscan_capped (memory guard)
  pipeline/      DatasetSpec, stage chain (stages.py), cache protocol, run_full_pipeline, TDAEngine facade
  adapters/      text: Document, chunking, TF-IDF/SVD, run_text_pipeline, processor registry
  viz/           optional renderers: layouts, 2-D, 3-D + galaxy, PNG
```

Dependencies point downwards only (adapters -> pipeline -> mapper/analysis -> lenses/core -> config).
Patterns: Strategy and Registry (lenses, document processors), chain of stages
with an abort hook (pipeline/stages.py), Protocol-based cache (pipeline/cache.py),
a single finding factory (core/types.py), Adapter (adapters/text.py) and a
Facade (`TDAEngine`) for applications that run many pipelines with one setup.

```python
from cx_tda_engine import TDAEngine, TDAConfig, DatasetSpec

engine = TDAEngine(config=TDAConfig(anomaly_top_k=50), spec=DatasetSpec(id_col="record_id"))
result = engine.run(df, feature_cols, lens_name="density")        # own cache, own config
graph = engine.mapper(df, feature_cols, lens_name="pca", anomaly_scores=result["combined_scores"])
```

## Methodology in one paragraph

Features are standardised; persistent homology (ripser, on a seeded PCA
sample) gives Betti numbers at a data-adaptive noise scale rather than raw
bar counts; DBSCAN with a k-NN-derived eps gives global clusters; Isolation
Forest blended with distance-to-centroid gives a per-record score in [0, 1];
the lens slices the data into overlapping intervals and local clustering
turns each slice into Mapper nodes linked by shared records. Node and group
tiers are *lifts* against the dataset's own anomaly rate, because absolute
cut-offs flag every large group.
