"""The RAG pipeline: ingest -> retrieve -> generate."""
from dataclasses import dataclass
from pathlib import Path

import joblib

from .chunker import chunk_document
from .config import DOCS_DIR, INDEX_PATH, TOP_K
from .llm import NOT_FOUND, get_generator
from .loader import Document, load_documents
from .retriever import Hit, make_retriever


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    generator: str

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(h.chunk.source for h in self.hits))


class RAGPipeline:
    def __init__(self, retriever="tfidf", generator=None):
        self.retriever = make_retriever(retriever) if isinstance(retriever, str) else retriever
        self.generator = generator or get_generator()

    # ---- indexing -------------------------------------------------------
    def ingest_documents(self, docs: list[Document]) -> tuple[int, int]:
        chunks = []
        for doc in docs:
            chunks.extend(chunk_document(doc, start_id=len(chunks)))
        if not chunks:
            raise ValueError("No text found to index.")
        self.retriever.fit(chunks)
        return len(docs), len(chunks)

    def ingest_folder(self, folder: Path = DOCS_DIR) -> tuple[int, int]:
        return self.ingest_documents(load_documents(folder))

    def save(self, path: Path = INDEX_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.retriever, path)

    @classmethod
    def load(cls, path: Path = INDEX_PATH, generator=None) -> "RAGPipeline":
        if not Path(path).exists():
            raise FileNotFoundError("No index found. Run `python -m docchat ingest` first.")
        return cls(retriever=joblib.load(path), generator=generator)

    # ---- querying -------------------------------------------------------
    def retrieve(self, question: str, k: int = TOP_K) -> list[Hit]:
        return self.retriever.search(question, k)

    def ask(self, question: str, k: int = TOP_K) -> Answer:
        hits = self.retrieve(question, k)
        if not hits or hits[0].score < self.retriever.min_score:
            return Answer(question, NOT_FOUND, [], self.generator.name)   # skip the LLM call
        return Answer(question, self.generator.generate(question, hits), hits,
                      self.generator.name)
