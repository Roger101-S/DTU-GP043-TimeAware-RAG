"""The two baselines the report compares against.

NaiveRAG        : single-pass top-K cosine similarity. Time-blind.
SoftTimeDecayRAG: similarity blended with exponential half-life decay (time as a SCORE, not validity):
                      score = alpha * sim_norm + (1 - alpha) * 0.5 ** (age_days / half_life)
                  Older notes are down-weighted but can still reach the generator. The half-life form is the
                  standard exponential-decay idea also used by temporal-rag (Emmimal, MIT) and similar
                  recency-weighted retrievers; this is our own small implementation of it.

Both are scoped to one patient (all pipelines are), and both treat "now" as the fixed REFERENCE_DATE
because they have no query-date understanding - that is exactly the limitation RQ1 probes.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .generator import Answerer, context_token_estimate
from .schema import PipelineResult, Query
from .temporal_filter import parse_query


class _BasePipeline:
    name = "base"

    def __init__(self, store, embedder, extractor, answerer: Answerer, meter, top_k: int, reference_date):
        self.store, self.embedder, self.extractor = store, embedder, extractor
        self.answerer, self.meter, self.top_k, self.reference_date = answerer, meter, top_k, reference_date

    def _finish(self, q: Query, parsed, retrieved, context, t0) -> PipelineResult:
        answer, state = self.answerer.answer(q.text, parsed.condition, parsed.as_of.date().isoformat(), context)
        return PipelineResult(
            query_id=q.query_id, pipeline=self.name, retrieved=retrieved, context=context,
            answer=answer, answer_state=state, suppressed=[],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            context_tokens=context_token_estimate(context), arbitration_tokens=0,
            llm_calls=0,       # filled in by the runner from the shared meter
        )


class NaiveRAG(_BasePipeline):
    name = "naive"

    def run(self, q: Query) -> PipelineResult:
        t0 = time.perf_counter()
        parsed = parse_query(q.text, self.extractor, self.reference_date)
        emb = self.embedder.encode([q.text])[0]
        hits = self.store.query(emb, q.patient_id, self.top_k)
        chunks = []
        for c, s in hits:
            c.score = s
            chunks.append(c)
        return self._finish(q, parsed, chunks, list(chunks), t0)


class SoftTimeDecayRAG(_BasePipeline):
    name = "decay"

    def __init__(self, *args, alpha: float = 0.6, half_life_days: float = 365.0, pool: int = 20, **kw):
        super().__init__(*args, **kw)
        self.alpha, self.half_life, self.pool = alpha, half_life_days, pool

    def run(self, q: Query) -> PipelineResult:
        t0 = time.perf_counter()
        parsed = parse_query(q.text, self.extractor, self.reference_date)
        emb = self.embedder.encode([q.text])[0]
        hits = self.store.query(emb, q.patient_id, self.pool)
        if not hits:
            return self._finish(q, parsed, [], [], t0)
        sims = np.array([s for _, s in hits], dtype=np.float64)
        span = sims.max() - sims.min()
        sim_norm = (sims - sims.min()) / span if span > 0 else np.ones_like(sims)   # min-max within the pool
        scored = []
        for (c, _), sn in zip(hits, sim_norm):
            age = max((self.reference_date - c.doc_date).days, 0)
            decay = 0.5 ** (age / self.half_life)
            c.score = float(self.alpha * sn + (1 - self.alpha) * decay)
            scored.append(c)
        scored.sort(key=lambda c: (-c.score, c.note_id))
        top = scored[: self.top_k]
        return self._finish(q, parsed, top, list(top), t0)
