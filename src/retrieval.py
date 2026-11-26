"""Retrieval / RAG layer.

Two interchangeable retrievers behind one interface:

* `TfidfRetriever`   -- pure-numpy, no model download, no network. Runs out of
                        the box so the whole pipeline (and CI) works offline.
* `EmbeddingRetriever` (optional) -- sentence-transformers embeddings + cosine
                        similarity. This is the "real" version.

Both chunk the markdown protocols by section, embed/score a query, and return
the top-k chunks with their source id for grounding/citation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

import numpy as np


@dataclass
class Chunk:
    source: str          # protocol id, e.g. "hypertension"
    text: str            # the chunk body
    score: float = 0.0   # filled in at query time


def load_protocol_chunks(protocols_dir: str | Path) -> List[Chunk]:
    """Split each protocol .md into section-level chunks (split on `## ` headings)."""
    protocols_dir = Path(protocols_dir)
    chunks: List[Chunk] = []
    for md in sorted(protocols_dir.glob("*.md")):
        source = md.stem
        text = md.read_text(encoding="utf-8")
        # Split on level-2 headings; keep the heading with its body.
        parts = re.split(r"\n(?=## )", text)
        for part in parts:
            body = part.strip()
            if body:
                chunks.append(Chunk(source=source, text=body))
    return chunks


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> List[Chunk]: ...


# ---------------------------------------------------------------------------
# TF-IDF retriever (default; no dependencies beyond numpy)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


class TfidfRetriever:
    """A tiny, transparent TF-IDF + cosine retriever.

    Deliberately simple so the retrieval mechanics are easy to read. It is the
    deterministic default used by tests and CI.
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.matrix: np.ndarray | None = None
        self._fit()

    def _fit(self) -> None:
        docs_tokens = [_tokenize(c.text) for c in self.chunks]
        for toks in docs_tokens:
            for t in set(toks):
                self.vocab.setdefault(t, len(self.vocab))

        n_docs = len(self.chunks)
        df = np.zeros(len(self.vocab))
        for toks in docs_tokens:
            for t in set(toks):
                df[self.vocab[t]] += 1
        # Smoothed inverse document frequency.
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0

        self.matrix = np.zeros((n_docs, len(self.vocab)))
        for i, toks in enumerate(docs_tokens):
            self.matrix[i] = self._vectorize(toks)

    def _vectorize(self, tokens: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for t in tokens:
            j = self.vocab.get(t)
            if j is not None:
                vec[j] += 1.0
        assert self.idf is not None
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def search(self, query: str, k: int = 3) -> List[Chunk]:
        assert self.matrix is not None
        q = self._vectorize(_tokenize(query))
        sims = self.matrix @ q  # rows are unit-normalized, q too -> cosine
        order = np.argsort(-sims)[:k]
        out: List[Chunk] = []
        for idx in order:
            if sims[idx] <= 0:
                continue
            c = self.chunks[idx]
            out.append(Chunk(source=c.source, text=c.text, score=float(sims[idx])))
        return out


# ---------------------------------------------------------------------------
# Embedding retriever (optional; the "real" version for Level 2)
# ---------------------------------------------------------------------------
class EmbeddingRetriever:
    """Cosine-similarity retrieval over sentence-transformer embeddings.

    Requires `pip install sentence-transformers` and a one-time model download
    (runs fine on Apple Silicon / MPS). Falls back loudly if not installed so
    you know to install it rather than silently degrading.
    """

    def __init__(self, chunks: List[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - exercised only when chosen
            raise ImportError(
                "EmbeddingRetriever needs sentence-transformers:\n"
                "    pip install sentence-transformers\n"
                "Or use TfidfRetriever, which needs no extra install."
            ) from e
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int = 3) -> List[Chunk]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        order = np.argsort(-sims)[:k]
        return [
            Chunk(source=self.chunks[i].source, text=self.chunks[i].text, score=float(sims[i]))
            for i in order
        ]

# Bragi adding
def split_into_windows(chunks, max_words=120, overlap_words=20):
    """Sub-divide each section chunk into overlapping fixed-size word windows.
    
    Words approximate tokens (~1 token ~ 0.75 words), so this needs no extra dependency.
    The section heading is prepended to continuation windows so each chunk stays self-describing
    (helps both retrieval and citation)."""

    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    step = max_words - overlap_words
    out = []
    for c in chunks:
        lines = c.text.splitlines()
        heading = lines[0].strip() if lines and lines[0].lstrip().startswith("#") else ""
        words = c.text.split()
        if len(words) <= max_words:
            out.append(c)
            continue
        start, first = 0, True
        while start < len(words):
            window = " ".join(words[start:start + max_words])
            if heading and not first and not window.startswith(heading):
                window = f"{heading} (cont.) {window}"
            out.append(Chunk(source=c.source, text=window))
            if start + max_words >= len(words):
                break
            start += step
            first = False
    return out

    
