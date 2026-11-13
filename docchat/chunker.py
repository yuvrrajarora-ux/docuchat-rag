"""Paragraph-aware chunking with overlap.

Paragraphs (and headings) are packed together until a size budget is reached, so a
heading stays attached to the text beneath it. Very long paragraphs fall back to a
sliding word window. The last paragraph(s) of a chunk are repeated at the start of
the next one so answers that straddle a boundary are not lost.
"""
import re
from dataclasses import dataclass

from .config import CHUNK_OVERLAP, CHUNK_WORDS
from .loader import Document


@dataclass
class Chunk:
    id: int
    source: str
    text: str


def _split_long(block: str, size: int, overlap: int) -> list[str]:
    words = block.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    pieces: list[str] = []
    for b in blocks:
        pieces.extend([b] if len(b.split()) <= size else _split_long(b, size, overlap))

    chunks: list[str] = []
    current: list[str] = []
    n_words = 0
    for piece in pieces:
        pw = len(piece.split())
        if current and n_words + pw > size:
            chunks.append("\n\n".join(current))
            carry, carried = [], 0
            for prev in reversed(current):
                w = len(prev.split())
                if carried + w > overlap:
                    break
                carry.insert(0, prev)
                carried += w
            current, n_words = carry, carried
        current.append(piece)
        n_words += pw
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(doc: Document, start_id: int = 0, **kwargs) -> list[Chunk]:
    return [Chunk(start_id + i, doc.source, t)
            for i, t in enumerate(chunk_text(doc.text, **kwargs))]
