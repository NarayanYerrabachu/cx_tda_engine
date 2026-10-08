"""Unstructured data: documents and free text become a TDA-ready table.

Each document (or chunk of one) is a row; its text is vectorised with TF-IDF,
optionally compressed with truncated SVD (latent semantic analysis), and the
resulting numeric columns are the features :func:`run_full_pipeline` runs on.
Metadata (file name, type, word count, parent file) rides along as display and
grouping columns.

File parsing (PDF, Word, e-mail, archives) is deliberately NOT here: it needs
heavy, format-specific libraries. Applications implement
:class:`DocumentProcessor` for their formats and hand the library
:class:`Document` objects; :func:`register_processor` / :func:`detect_processor`
give them a registry keyed by file extension.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TDAConfig, resolve
from .pipeline import DatasetSpec, run_full_pipeline
from .types import PipelineResult

TFIDF_PREFIX = "tfidf_"
SVD_PREFIX = "lsa_"
ID_COL = "record_id"
META_COLS = ("file_name", "file_type", "word_count", "char_count", "parent_file")

# Chunking: used when a corpus has too few documents for TDA (e.g. one PDF).
CHUNK_TARGET = 12        # aim for roughly this many chunks per document
CHUNK_MIN_WORDS = 30
CHUNK_MAX_WORDS = 250


@dataclass
class Document:
    """One extracted document: text plus provenance."""

    source_file: str                    # original file as received
    file_name: str                      # name of this specific document (unique per corpus)
    file_type: str                      # extension without dot: "pdf", "docx", "eml", "txt"
    content: str                        # extracted text
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_file: str | None = None      # set when extracted from an archive, e-mail or chunked


class DocumentProcessor(ABC):
    """Turns a file into :class:`Document` objects. Implemented by applications."""

    supported_extensions: list[str] = []

    def can_process(self, file_path: str) -> bool:
        if "." not in file_path:
            return False
        return file_path.rsplit(".", 1)[-1].lower() in self.supported_extensions

    @abstractmethod
    def process(self, file_path: str, source_path: str = "") -> list[Document]:
        """Extract every document from ``file_path`` (``source_path`` = original upload name)."""


_PROCESSORS: dict[str, DocumentProcessor] = {}


def register_processor(processor: DocumentProcessor) -> None:
    for ext in processor.supported_extensions:
        _PROCESSORS[ext.lower()] = processor


def detect_processor(extension: str) -> DocumentProcessor | None:
    return _PROCESSORS.get((extension or "").lstrip(".").lower())


def supported_extensions() -> set[str]:
    return set(_PROCESSORS)


def documents_from_texts(texts: Sequence[str], ids: Sequence[str] | None = None,
                         file_type: str = "txt", labels: Sequence[str] | None = None) -> list[Document]:
    """Wrap plain strings as documents (ids default to ``doc-0001`` ...; ``labels`` go to metadata)."""
    if ids is not None and len(ids) != len(texts):
        raise ValueError("ids must have the same length as texts")
    if labels is not None and len(labels) != len(texts):
        raise ValueError("labels must have the same length as texts")
    out = []
    for i, t in enumerate(texts):
        name = str(ids[i]) if ids is not None else f"doc-{i + 1:04d}"
        meta = {"label": str(labels[i])} if labels is not None else {}
        out.append(Document(source_file=name, file_name=name, file_type=file_type, content=t or "", metadata=meta))
    return out


def chunk_documents(docs: Iterable[Document], min_rows: int) -> list[Document]:
    """Split documents into word-window chunks when there are fewer than ``min_rows``.

    TDA needs several rows; a single PDF would otherwise give one row and
    fail. Each chunk is its own Document named ``<original>_chunk-NNN`` with
    ``parent_file`` pointing at the original. Corpora that already have enough
    documents are returned unchanged.
    """
    docs = list(docs)
    if len(docs) >= min_rows:
        return docs
    out: list[Document] = []
    for doc in docs:
        words = (doc.content or "").split()
        if not words:
            out.append(doc)
            continue
        size = max(CHUNK_MIN_WORDS, min(CHUNK_MAX_WORDS, len(words) // CHUNK_TARGET or 1))
        n_chunks = (len(words) + size - 1) // size
        if n_chunks <= 1:
            out.append(doc)
            continue
        for i in range(n_chunks):
            out.append(Document(
                source_file=doc.source_file,
                file_name=f"{doc.file_name}_chunk-{i + 1:03d}",
                file_type=doc.file_type,
                content=" ".join(words[i * size:(i + 1) * size]),
                metadata={**doc.metadata, "chunk_index": i + 1, "chunk_count": n_chunks, "chunk_words": size},
                parent_file=doc.parent_file or doc.file_name,
            ))
    return out


def documents_to_dataframe(
    docs: Sequence[Document],
    max_features: int = 200,
    n_components: int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: str | list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """One row per document: metadata columns plus numeric text features.

    Text features are TF-IDF columns ``tfidf_<term>`` (sublinear tf, unicode
    accents stripped, pure tf for a single document). With ``n_components``
    the TF-IDF matrix is compressed by truncated SVD into ``lsa_<k>`` columns,
    which gives a dense, low-dimensional space that Mapper and persistent
    homology handle far better than 200 sparse term columns.

    Returns ``(df, feature_cols)``; ``feature_cols`` lists only the numeric
    text features. Metadata keys of the documents are added as columns too.
    """
    if not docs:
        raise ValueError("documents_to_dataframe requires at least one Document.")
    texts = [d.content or "" for d in docs]
    meta_rows = [{
        ID_COL:        d.file_name,
        "file_name":   d.file_name,
        "file_type":   d.file_type,
        "word_count":  len(t.split()),
        "char_count":  len(t),
        "parent_file": d.parent_file,
        **{k: v for k, v in d.metadata.items() if k not in META_COLS and k != ID_COL},
    } for d, t in zip(docs, texts)]

    vectorizer = TfidfVectorizer(max_features=max_features, use_idf=len(docs) >= 2, sublinear_tf=True,
                                 strip_accents="unicode", min_df=1, ngram_range=ngram_range,
                                 stop_words=stop_words)
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError as exc:  # only stop words / empty texts
        raise ValueError("No usable terms in the documents (empty texts or only stop words).") from exc

    if n_components is not None and n_components > 0:
        k = max(1, min(n_components, tfidf.shape[1] - 1, len(docs) - 1))
        X = TruncatedSVD(n_components=k, random_state=seed).fit_transform(tfidf)
        feature_cols = [f"{SVD_PREFIX}{i + 1}" for i in range(k)]
    else:
        X = tfidf.toarray()
        feature_cols = [f"{TFIDF_PREFIX}{term}" for term in vectorizer.get_feature_names_out()]

    feat_df = pd.DataFrame(np.asarray(X, dtype=float), columns=feature_cols)
    df = pd.concat([pd.DataFrame(meta_rows).reset_index(drop=True), feat_df], axis=1)
    return df, feature_cols


TEXT_SPEC = DatasetSpec(
    id_col=ID_COL,
    group_cols=("file_type", "parent_file"),
    display_cols=("file_name", "file_type", "word_count", "parent_file"),
    stat_cols=("word_count",),
)


def run_text_pipeline(
    docs: Sequence[Document] | Sequence[str],
    lens_name: str = "pca",
    n_intervals: int = 10,
    overlap: float = 0.5,
    *,
    max_features: int = 200,
    n_components: int | None = 20,
    chunk: bool = True,
    spec: DatasetSpec | None = None,
    config: TDAConfig | None = None,
    **pipeline_kwargs: Any,
) -> tuple[PipelineResult, pd.DataFrame, list[str]]:
    """Documents (or raw strings) straight to the pipeline result.

    Chunks the corpus when it has fewer than ``config.min_rows`` documents,
    vectorises it (TF-IDF, SVD to ``n_components`` by default) and runs
    :func:`run_full_pipeline` with :data:`TEXT_SPEC` unless ``spec`` is given.
    Returns ``(result, df, feature_cols)`` so the caller keeps the table the
    findings refer to.
    """
    cfg = resolve(config)
    documents: list[Document]
    if docs and isinstance(docs[0], str):
        documents = documents_from_texts([str(t) for t in docs])
    else:
        documents = [d for d in docs if isinstance(d, Document)]
    if chunk:
        documents = chunk_documents(documents, cfg.min_rows)
    df, feature_cols = documents_to_dataframe(documents, max_features=max_features,
                                              n_components=n_components, seed=cfg.seed)
    result = run_full_pipeline(df, feature_cols, lens_name, n_intervals, overlap,
                               spec=spec or TEXT_SPEC, config=cfg, **pipeline_kwargs)
    return result, df, feature_cols
