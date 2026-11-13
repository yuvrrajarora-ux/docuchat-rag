"""Read .txt / .md / .pdf files into Document objects."""
from dataclasses import dataclass
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS


@dataclass
class Document:
    source: str   # file name, used for citations
    text: str


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("PDF support needs `pip install pypdf`") from e
    return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def read_file(path: Path) -> Document:
    if path.suffix.lower() == ".pdf":
        text = read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(source=path.name, text=text)


def load_documents(folder: Path) -> list[Document]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    docs = [read_file(p) for p in files]
    return [d for d in docs if d.text.strip()]
