"""Retrieval evaluation: does the right document show up in the top-k results?

Metrics: hit@1, hit@3 and MRR (mean reciprocal rank) over data/eval/qa.json.
"""
import json
from pathlib import Path

from .config import DOCS_DIR, EVAL_PATH, REPORT_PATH
from .rag import RAGPipeline


def evaluate_retrieval(pipeline: RAGPipeline, qa_path: Path = EVAL_PATH, k: int = 5) -> dict:
    qa = json.loads(Path(qa_path).read_text())
    ranks, misses = [], []
    for item in qa:
        sources = [h.chunk.source for h in pipeline.retrieve(item["question"], k)]
        rank = sources.index(item["source"]) + 1 if item["source"] in sources else None
        ranks.append(rank)
        if rank != 1:
            misses.append({"question": item["question"], "expected": item["source"],
                           "got": sources[:3]})
    n = len(qa)
    return {
        "retriever": pipeline.retriever.name,
        "n_questions": n,
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "misses": misses,
    }


def run(kinds=("tfidf",)) -> list[dict]:
    results = []
    for kind in kinds:
        pipe = RAGPipeline(retriever=kind)
        pipe.ingest_folder(DOCS_DIR)
        results.append(evaluate_retrieval(pipe))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))
    return results
