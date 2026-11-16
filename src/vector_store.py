"""Vector stores holding embeddings + HARD temporal metadata.

ChromaStore  : ChromaDB (as in the report). Patient and date filters are applied INSIDE the database
               via metadata `where` clauses, i.e. the filter is a hard constraint, not a re-ranking score.
NumpyStore   : brute-force fallback with the same interface (used if chromadb is not installed).
"""
from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from .schema import Chunk


def _ts(dt) -> int:
    return int(dt.timestamp())


class NumpyStore:
    name = "numpy"

    def __init__(self):
        self._chunks: list = []
        self._emb: Optional[np.ndarray] = None

    def add(self, chunks: list, embeddings: np.ndarray) -> None:
        self._chunks.extend(chunks)
        self._emb = embeddings if self._emb is None else np.vstack([self._emb, embeddings])

    def query(self, embedding: np.ndarray, patient_id: str, k: int,
              max_ts: Optional[int] = None, min_ts: Optional[int] = None) -> list:
        if self._emb is None:
            return []
        sims = self._emb @ embedding
        hits = []
        for i, c in enumerate(self._chunks):
            if c.patient_id != patient_id:
                continue
            ts = _ts(c.doc_date)
            if max_ts is not None and ts > max_ts:
                continue
            if min_ts is not None and ts < min_ts:
                continue
            hits.append((c, float(sims[i])))
        hits.sort(key=lambda x: (-x[1], x[0].note_id))
        return hits[:k]


class ChromaStore:
    name = "chroma"

    def __init__(self, collection_name: Optional[str] = None):
        import chromadb
        self._client = chromadb.EphemeralClient()
        self._col = self._client.create_collection(
            name=collection_name or f"notes_{uuid.uuid4().hex[:10]}",
            metadata={"hnsw:space": "cosine"},
        )
        self._by_id: dict = {}

    def add(self, chunks: list, embeddings: np.ndarray) -> None:
        for c in chunks:
            self._by_id[c.note_id] = c
        self._col.add(
            ids=[c.note_id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{"patient_id": c.patient_id, "note_id": c.note_id,
                        "date_ts": _ts(c.doc_date), "date_iso": c.doc_date.date().isoformat(),
                        "date_source": c.date_source} for c in chunks],
        )

    def query(self, embedding: np.ndarray, patient_id: str, k: int,
              max_ts: Optional[int] = None, min_ts: Optional[int] = None) -> list:
        clauses = [{"patient_id": {"$eq": patient_id}}]
        if max_ts is not None:
            clauses.append({"date_ts": {"$lte": int(max_ts)}})
        if min_ts is not None:
            clauses.append({"date_ts": {"$gte": int(min_ts)}})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        n = min(k, self._col.count())
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[embedding.tolist()], n_results=n, where=where)
        out = []
        for note_id, dist in zip(res["ids"][0], res["distances"][0]):
            out.append((self._by_id[note_id], 1.0 - float(dist)))     # cosine distance -> similarity
        return out


def make_store(prefer: str = "chroma"):
    if prefer == "chroma":
        try:
            return ChromaStore()
        except Exception:        # chromadb missing or incompatible
            return NumpyStore()
    return NumpyStore()
