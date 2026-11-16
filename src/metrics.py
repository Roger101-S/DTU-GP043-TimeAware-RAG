"""Metrics for RQ1-RQ3. Gold labels are read ONLY here; pipelines never see them.

Per query (gold = ground truth for the queried condition as of the query date):
  wrong-time notes   = superseded notes (older, different state) + future notes (dated after the as-of date)
  evidence_recall    : the note that states the correct state is in the final context               (RQ1)
  stale_rate         : share of final-context notes that are wrong-time                              (RQ1)
  clean_context      : correct note present AND no wrong-time note present. LLM-independent proxy
                       for "a reader can get the current state right"                                (RQ1/RQ2)
  exposure           : correct note AND a wrong-time note are both in the final context, i.e. the
                       generator would see contradictory evidence                                    (RQ2)
  conflict_present   : the pre-arbitration retrieved set contained the correct note + a wrong-time note
  conflict_resolved  : conflict_present AND clean_context after the pipeline ran
  accuracy           : the answer's state label equals the gold state (reader = LLM or rule proxy)
  contradictory      : the (rule) reader saw several states and returned CONFLICT
Cost (RQ3): latency_ms, context_tokens (to the generator), arbitration_tokens (LLM inside the arbitrator),
llm_calls.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def score_query(gold: dict, res) -> dict:
    gold_id = gold["gold_note_id"]
    wrong = set(gold["superseded_note_ids"]) | set(gold["future_note_ids"])
    ctx = [c.note_id for c in res.context]
    ret = {c.note_id for c in res.retrieved}
    ctx_set = set(ctx)
    gold_in_ctx = gold_id in ctx_set
    wrong_in_ctx = wrong & ctx_set
    clean = gold_in_ctx and not wrong_in_ctx
    present = (gold_id in ret) and bool(wrong & ret)
    return {
        "query_id": gold["query_id"], "pipeline": res.pipeline, "kind": gold["kind"],
        "evidence_recall": float(gold_in_ctx),
        "stale_rate": (len(wrong_in_ctx) / len(ctx)) if ctx else 0.0,
        "clean_context": float(clean),
        "exposure": float(gold_in_ctx and bool(wrong_in_ctx)),
        "conflict_present": float(present),
        "conflict_resolved": float(present and clean),
        "accuracy": float(res.answer_state == gold["gold_state"]),
        "contradictory": float(res.answer_state == "CONFLICT"),
        "latency_ms": res.latency_ms, "context_tokens": res.context_tokens,
        "arbitration_tokens": res.arbitration_tokens, "llm_calls": res.llm_calls,
        "n_suppressed": len(res.suppressed),
    }


def aggregate(rows: list) -> dict:
    """Mean metrics over a list of per-query rows (all from one pipeline)."""
    def mean(k):
        return float(np.mean([r[k] for r in rows])) if rows else float("nan")
    present = sum(r["conflict_present"] for r in rows)
    resolved = sum(r["conflict_resolved"] for r in rows)
    return {
        "n_queries": len(rows),
        "accuracy": mean("accuracy"), "clean_context": mean("clean_context"),
        "evidence_recall": mean("evidence_recall"), "stale_rate": mean("stale_rate"),
        "exposure": mean("exposure"), "contradictory": mean("contradictory"),
        "conflict_resolution_rate": (resolved / present) if present else float("nan"),
        "n_conflicts": int(present),
        "latency_ms": mean("latency_ms"),
        "latency_ms_median": float(np.median([r["latency_ms"] for r in rows])) if rows else float("nan"),
        "context_tokens": mean("context_tokens"), "arbitration_tokens": mean("arbitration_tokens"),
        "llm_calls": mean("llm_calls"),
    }


def paired_bootstrap(a: list, b: list, n_boot: int = 2000, seed: int = 0) -> Optional[tuple]:
    """Mean(a - b) with a 95% percentile bootstrap CI, resampling queries (paired). None if empty."""
    if not a or len(a) != len(b):
        return None
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"
