import numpy as np
import pytest

from cx_tda_engine import text
from cx_tda_engine.text import (
    Document,
    DocumentProcessor,
    chunk_documents,
    detect_processor,
    documents_from_texts,
    documents_to_dataframe,
    register_processor,
    run_text_pipeline,
)

TOPICS = {
    "finance": "budget invoice payment ledger audit revenue cost margin tax fiscal quarter balance",
    "medical": "patient clinic diagnosis treatment nurse hospital symptom dosage therapy ward",
    "legal":   "contract clause liability court statute plaintiff verdict appeal counsel ruling",
}


def corpus(n_per_topic: int = 30, seed: int = 0) -> list[Document]:
    rng = np.random.default_rng(seed)
    docs = []
    for topic, words in TOPICS.items():
        vocab = words.split()
        for i in range(n_per_topic):
            body = " ".join(rng.choice(vocab, 40))
            docs.append(Document(source_file=f"{topic}.zip", file_name=f"{topic}-{i:03d}", file_type="txt",
                                 content=body, metadata={"topic": topic}, parent_file=f"{topic}.zip"))
    return docs


def test_documents_to_dataframe_tfidf_and_metadata():
    docs = corpus(5)
    df, cols = documents_to_dataframe(docs)
    assert len(df) == 15 and all(c.startswith("tfidf_") for c in cols)
    assert {"record_id", "file_name", "file_type", "word_count", "char_count", "parent_file", "topic"} <= set(df)
    assert df["record_id"].tolist() == [d.file_name for d in docs]
    assert (df["word_count"] == 40).all()
    assert set(cols) == {f"tfidf_{w}" for t in TOPICS.values() for w in t.split()}


def test_documents_to_dataframe_svd():
    df, cols = documents_to_dataframe(corpus(5), n_components=4)
    assert cols == ["lsa_1", "lsa_2", "lsa_3", "lsa_4"]
    assert not df[cols].isna().any().any()
    _, few = documents_to_dataframe(corpus(1), n_components=10)     # capped at n_docs - 1
    assert len(few) == 2


def test_documents_to_dataframe_errors():
    with pytest.raises(ValueError):
        documents_to_dataframe([])
    with pytest.raises(ValueError, match="No usable terms"):
        documents_to_dataframe(documents_from_texts(["", "   "]))


def test_single_document_uses_pure_tf():
    df, cols = documents_to_dataframe(documents_from_texts(["alpha beta beta gamma"]))
    assert len(df) == 1 and len(cols) == 3


def test_chunk_documents_only_when_too_few():
    long = Document("a.pdf", "a.pdf", "pdf", " ".join(f"w{i}" for i in range(600)))
    chunks = chunk_documents([long], min_rows=5)
    assert len(chunks) == 12
    assert chunks[0].file_name == "a.pdf_chunk-001" and chunks[0].parent_file == "a.pdf"
    assert chunks[0].metadata["chunk_count"] == 12
    assert " ".join(c.content for c in chunks) == long.content
    many = documents_from_texts(["x y"] * 6)
    assert chunk_documents(many, min_rows=5) == many
    short = Document("s.txt", "s.txt", "txt", "only three words")
    assert chunk_documents([short], min_rows=5) == [short]


def test_documents_from_texts_ids_labels():
    docs = documents_from_texts(["a b", "c d"], ids=["x", "y"], labels=["L1", "L2"])
    assert [d.file_name for d in docs] == ["x", "y"] and docs[1].metadata == {"label": "L2"}
    with pytest.raises(ValueError):
        documents_from_texts(["a"], ids=["x", "y"])


def test_run_text_pipeline_separates_topics():
    result, df, cols = run_text_pipeline(corpus(30), n_components=10)
    assert "error" not in result
    assert result["meta"]["n_docs"] == 90 and cols[0] == "lsa_1"
    assert result["meta"]["category_features"] == ["file_type", "parent_file"]
    # the three topics are far apart: at least three clusters, and every
    # Mapper node is topic-pure
    topic_of = dict(zip(df["record_id"], df["topic"]))
    for node in result["graph"]["mapper_nodes"]:
        assert len({topic_of[s] for s in node["sources"]}) == 1
    assert result["meta"]["n_clusters"] >= 3
    assert result["anomalies"][0]["extra"]["file_name"]


def test_run_text_pipeline_from_strings_and_chunking():
    one_long = [" ".join(np.random.default_rng(1).choice(TOPICS["legal"].split(), 900))]
    result, df, _ = run_text_pipeline(one_long, n_components=5)
    assert "error" not in result and len(df) >= 5
    assert df["parent_file"].nunique() == 1 and df["file_name"].str.contains("_chunk-").all()


def test_processor_registry():
    class Txt(DocumentProcessor):
        supported_extensions = ["txt", "md"]

        def process(self, file_path, source_path=""):
            return [Document(source_path or file_path, file_path, "txt", "hello world")]

    register_processor(Txt())
    try:
        assert detect_processor(".TXT") is detect_processor("md")
        assert detect_processor("pdf") is None
        assert {"txt", "md"} <= text.supported_extensions()
        assert detect_processor("txt").can_process("notes.txt") and not detect_processor("txt").can_process("x")
        assert detect_processor("txt").process("notes.txt")[0].content == "hello world"
    finally:
        text._PROCESSORS.clear()
