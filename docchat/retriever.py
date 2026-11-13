"""Retrievers: TF-IDF (default, offline) and dense embeddings (optional)."""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunker import Chunk
from .config import EMBEDDING_MODEL


@dataclass
class Hit:
    chunk: Chunk
    score: float


class TfidfRetriever:
    name = "tfidf"
    min_score = 0.05   # below this the question is treated as "not in the documents"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.chunks: list[Chunk] = []
        self.matrix = None

    def fit(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])
        return self

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()   # tf-idf rows are L2-normalised
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class EmbeddingRetriever:
    name = "embeddings"
    min_score = 0.25

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self.chunks: list[Chunk] = []
        self.matrix = None

    def _encoder(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("Embeddings need `pip install sentence-transformers`") from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __getstate__(self):                      # keep the model out of the saved index
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def fit(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.matrix = self._encoder().encode([c.text for c in chunks],
                                             normalize_embeddings=True)
        return self

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self._encoder().encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def make_retriever(kind: str = "tfidf"):
    if kind == "tfidf":
        return TfidfRetriever()
    if kind in ("embeddings", "dense"):
        return EmbeddingRetriever()
    raise ValueError(f"Unknown retriever '{kind}' (use 'tfidf' or 'embeddings')")
