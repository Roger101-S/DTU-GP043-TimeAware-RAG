"""Builds the index and the pipelines once, and measures LLM usage per query."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .arbitrator import Arbitrator
from .arbitrator_graph import TimeAwareAgenticRAG
from .baselines import NaiveRAG, SoftTimeDecayRAG
from .config import Config
from .embedder import get_embedder
from .generator import Answerer
from .llm import TokenMeter, make_llm
from .ner_ingestion import DateExtractor, build_chunks
from .schema import Note, Query
from .vector_store import make_store

log = logging.getLogger(__name__)
ALL_PIPELINES = ["naive", "decay", "filter", "arbitrator", "full"]


def load_notes(path: str) -> list:
    notes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                notes.append(Note(d["note_id"], d["patient_id"], d["text"], d.get("filed_date")))
    return notes


def load_queries(path: str, split: Optional[str] = None) -> list:
    """Returns list of dicts (query fields + gold labels). Gold labels are used ONLY by metrics.py."""
    with open(path, encoding="utf-8") as f:
        qs = json.load(f)
    return [q for q in qs if split in (None, "all") or q["split"] == split]


def build_system(cfg: Config, notes: list, llm=None, extractor: Optional[DateExtractor] = None,
                 embedder=None, store=None):
    """Ingest notes (NER -> metadata -> embeddings -> vector DB) and return (pipelines, meter, report)."""
    extractor = extractor or DateExtractor(cfg.date_order)
    chunks, report = build_chunks(notes, extractor)
    embedder = embedder or get_embedder(cfg.embedder)
    embedder.fit([c.text for c in chunks])
    store = store or make_store("chroma")
    store.add(chunks, embedder.encode([c.text for c in chunks]))
    report["embedder"], report["store"] = embedder.name, store.name

    meter = TokenMeter()
    answerer = Answerer(llm=llm, meter=meter)
    arbitrator = Arbitrator(cfg.arbitrator_mode, llm=llm, meter=meter)
    report["arbitrator_mode"] = arbitrator.mode
    report["reader"] = answerer.reader

    common = dict(store=store, embedder=embedder, extractor=extractor, answerer=answerer, meter=meter,
                  top_k=cfg.top_k, reference_date=cfg.reference_date)
    pipelines = {
        "naive": NaiveRAG(**common),
        "decay": SoftTimeDecayRAG(**common, alpha=cfg.alpha, half_life_days=cfg.half_life_days),
    }
    graph_args = dict(store=store, embedder=embedder, extractor=extractor, answerer=answerer,
                      arbitrator=arbitrator, top_k=cfg.top_k, reference_date=cfg.reference_date)
    pipelines["filter"] = TimeAwareAgenticRAG(**graph_args, use_filter=True, use_arbitrator=False)
    pipelines["arbitrator"] = TimeAwareAgenticRAG(**graph_args, use_filter=False, use_arbitrator=True)
    pipelines["full"] = TimeAwareAgenticRAG(**graph_args, use_filter=True, use_arbitrator=True)
    return pipelines, meter, report


def run_measured(pipeline, query: Query, meter: TokenMeter):
    """Run one query and attach LLM call/token usage (delta of the shared meter)."""
    calls0, arb0 = meter.calls, meter.by_role.get("arbitrator", 0)
    res = pipeline.run(query)
    res.llm_calls = meter.calls - calls0
    res.arbitration_tokens = meter.by_role.get("arbitrator", 0) - arb0
    return res
