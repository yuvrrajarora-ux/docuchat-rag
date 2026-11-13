"""Paths and tunable settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "data" / "eval" / "qa.json"
INDEX_PATH = ROOT / "storage" / "index.joblib"
REPORT_PATH = ROOT / "reports" / "retrieval_eval.json"

CHUNK_WORDS = 110      # target chunk size
CHUNK_OVERLAP = 25     # words carried between neighbouring chunks
TOP_K = 4              # passages passed to the generator

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
