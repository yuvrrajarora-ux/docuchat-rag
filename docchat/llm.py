"""Answer generators.

* ClaudeGenerator     - used automatically when ANTHROPIC_API_KEY is set.
* ExtractiveGenerator - offline fallback: returns the most relevant sentences.
Both return text with [n] citations that map to the retrieved passages.
"""
import os
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import CLAUDE_MODEL
from .retriever import Hit

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the numbered context "
    "passages provided. Cite the passages you use like [1] or [2]. If the answer is not "
    "in the context, say you could not find it in the documents. Be concise."
)
NOT_FOUND = "I couldn't find that in the documents."


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{i}] (source: {h.chunk.source})\n{h.chunk.text}"
                          for i, h in enumerate(hits, 1))
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS}


class ExtractiveGenerator:
    name = "extractive (offline)"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> str:
        q = _tokens(question)
        scored = []
        for rank, hit in enumerate(hits, 1):
            for sent in re.split(r"(?<=[.!?])\s+|\n+", hit.chunk.text):
                clean = re.sub(r"^[#\-*\d.\s]+", "", sent).strip()
                if len(clean.split()) < 4:
                    continue
                overlap = len(q & _tokens(clean))
                if overlap:
                    scored.append((overlap, -rank, clean, rank))
        if not scored:
            return NOT_FOUND
        top = max(s[0] for s in scored)
        # keep only the best-matching sentences (ties allowed) so weak matches are dropped
        best = sorted((s for s in scored if s[0] == top), reverse=True)[: self.max_sentences]
        return " ".join(f"{clean} [{rank}]" for _, _, clean, rank in best)


class ClaudeGenerator:
    def __init__(self, model: str = CLAUDE_MODEL):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"Claude ({model})"

    def generate(self, question: str, hits: list[Hit]) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=500, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, hits)}])
        return msg.content[0].text.strip()


def get_generator():
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return ClaudeGenerator()
        except ImportError:
            print("ANTHROPIC_API_KEY set but `anthropic` is not installed; using offline mode.")
    return ExtractiveGenerator()
