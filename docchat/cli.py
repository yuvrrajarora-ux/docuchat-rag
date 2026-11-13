"""Command line interface.

  python -m docchat ingest [--docs data/docs] [--retriever tfidf|embeddings]
  python -m docchat ask "your question" [-k 4]
  python -m docchat chat
  python -m docchat eval [--retriever tfidf embeddings]
"""
import argparse

from .config import DOCS_DIR, TOP_K
from .rag import Answer, RAGPipeline


def show(ans: Answer, verbose: bool = True):
    print(f"\n{ans.text}\n")
    if ans.hits:
        print(f"Generator: {ans.generator}")
        for i, h in enumerate(ans.hits, 1):
            preview = " ".join(h.chunk.text.split())[:90]
            print(f"  [{i}] {h.chunk.source}  (score {h.score:.2f})  {preview}...")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(prog="docchat", description="Chat with your documents (RAG)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="build the search index")
    s.add_argument("--docs", default=str(DOCS_DIR))
    s.add_argument("--retriever", default="tfidf", choices=["tfidf", "embeddings"])

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=TOP_K)

    sub.add_parser("chat", help="interactive question loop")

    s = sub.add_parser("eval", help="evaluate retrieval quality")
    s.add_argument("--retriever", nargs="+", default=["tfidf"], choices=["tfidf", "embeddings"])

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        pipe = RAGPipeline(retriever=args.retriever)
        n_docs, n_chunks = pipe.ingest_folder(args.docs)
        pipe.save()
        print(f"Indexed {n_docs} documents into {n_chunks} chunks ({args.retriever}).")
    elif args.cmd == "ask":
        show(RAGPipeline.load().ask(args.question, args.k))
    elif args.cmd == "chat":
        pipe = RAGPipeline.load()
        print("Ask a question (blank line to quit).")
        while (q := input("> ").strip()):
            show(pipe.ask(q))
    elif args.cmd == "eval":
        from .evaluate import run
        for r in run(args.retriever):
            print(f"{r['retriever']:<11} n={r['n_questions']}  hit@1={r['hit@1']:.2f}  "
                  f"hit@3={r['hit@3']:.2f}  MRR={r['mrr']:.3f}")
            for m in r["misses"]:
                print(f"   miss: {m['question']!r} expected {m['expected']}, got {m['got']}")
