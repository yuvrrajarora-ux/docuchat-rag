import pytest

from docchat.chunker import chunk_text
from docchat.config import DOCS_DIR
from docchat.evaluate import evaluate_retrieval
from docchat.llm import NOT_FOUND, ExtractiveGenerator
from docchat.loader import Document, load_documents
from docchat.rag import RAGPipeline


@pytest.fixture(scope="module")
def pipe():
    p = RAGPipeline(retriever="tfidf", generator=ExtractiveGenerator())
    p.ingest_folder(DOCS_DIR)
    return p


def test_loader_reads_sample_docs():
    docs = load_documents(DOCS_DIR)
    assert len(docs) == 5 and all(d.text.strip() for d in docs)


def test_chunker_respects_size_and_keeps_all_text():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, size=80, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 80 + 20 for c in chunks)
    assert all(f"Paragraph {i}" in " ".join(chunks) for i in range(10))


def test_chunker_splits_one_huge_paragraph():
    chunks = chunk_text("word " * 500, size=100, overlap=10)
    assert len(chunks) >= 5


def test_retrieval_finds_right_document(pipe):
    hits = pipe.retrieve("What is the hotel cap per night?", k=3)
    assert hits[0].chunk.source == "expenses.md"


def test_answer_contains_fact_and_citation(pipe):
    ans = pipe.ask("How many days of annual leave do I get?")
    assert "24" in ans.text and "[1]" in ans.text
    assert "sick" not in ans.text
    assert "leave_policy.md" in ans.sources


def test_off_topic_question_is_declined(pipe):
    ans = pipe.ask("Who won the cricket world cup in 2011?")
    assert ans.text == NOT_FOUND and ans.hits == []


def test_save_and_load_roundtrip(pipe, tmp_path):
    path = tmp_path / "index.joblib"
    pipe.save(path)
    loaded = RAGPipeline.load(path, generator=ExtractiveGenerator())
    assert loaded.retrieve("probation period", 1)[0].chunk.source == "onboarding.md"


def test_retrieval_quality_floor(pipe):
    r = evaluate_retrieval(pipe)
    assert r["hit@3"] >= 0.85 and r["mrr"] >= 0.75


def test_empty_input_raises():
    with pytest.raises(ValueError):
        RAGPipeline(generator=ExtractiveGenerator()).ingest_documents([Document("a.txt", "")])
