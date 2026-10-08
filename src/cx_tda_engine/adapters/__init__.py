"""Adapters turn other kinds of input into the DataFrame + feature columns the pipeline takes.

Today: unstructured text (:mod:`text`). Future adapters (time series windows,
graphs, embeddings) follow the same shape: ``(df, feature_cols)`` out, a
matching :class:`~cx_tda_engine.pipeline.DatasetSpec` alongside.
"""
from .text import (
    TEXT_SPEC,
    Document,
    DocumentProcessor,
    chunk_documents,
    detect_processor,
    documents_from_texts,
    documents_to_dataframe,
    register_processor,
    run_text_pipeline,
    supported_extensions,
)

__all__ = ["TEXT_SPEC", "Document", "DocumentProcessor", "chunk_documents", "detect_processor",
           "documents_from_texts", "documents_to_dataframe", "register_processor", "run_text_pipeline",
           "supported_extensions"]
