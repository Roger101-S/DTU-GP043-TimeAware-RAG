"""Embedding back-ends: MiniLM (as in the report) and a dependency-free TF-IDF fallback."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


# Words that appear in nearly every query/note and carry no retrieval signal. Without this list the fallback
# embedder ranks short routine notes above condition notes (measured in our demo), because "patient", "status",
# "current" ... dominate a short query. Patient ids (P001) are constant inside a patient-scoped search.
_STOP = {"what", "is", "the", "of", "for", "as", "was", "a", "an", "and", "or", "in", "on", "to", "by", "with",
         "patient", "status", "current", "currently", "latest", "recorded", "year", "old", "male", "female",
         "visit", "clinic", "seen", "date", "consultation"}


class TfidfEmbedder:
    """Deterministic TF-IDF vectors (L2-normalised). No downloads; used for quick tests."""

    name = "tfidf"

    def __init__(self):
        self.vocab: dict = {}
        self.idf: np.ndarray | None = None

    @staticmethod
    def _tok(text: str) -> list:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in toks if t not in _STOP and not re.fullmatch(r"p\d{3,}", t)]

    def fit(self, texts: list) -> "TfidfEmbedder":
        df: Counter = Counter()
        for t in texts:
            df.update(set(self._tok(t)))
        terms = sorted(df)
        self.vocab = {w: i for i, w in enumerate(terms)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in terms], dtype=np.float32)
        return self

    def encode(self, texts: list) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode().")
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w, c in Counter(self._tok(t)).items():
                j = self.vocab.get(w)
                if j is not None:
                    out[r, j] = c * self.idf[j]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """sentence-transformers/all-MiniLM-L6-v2 (needs `sentence-transformers` and a one-time model download)."""

    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("sentence-transformers is not installed. Install it or set EMBEDDER=tfidf.") from e
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list) -> "SentenceTransformerEmbedder":
        return self  # pretrained: nothing to fit

    def encode(self, texts: list) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)


def get_embedder(name: str):
    name = (name or "minilm").lower()
    if name == "tfidf":
        return TfidfEmbedder()
    if name in ("minilm", "sentence-transformers", "st"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder '{name}'. Use 'minilm' or 'tfidf'.")
