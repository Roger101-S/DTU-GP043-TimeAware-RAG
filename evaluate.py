#!/usr/bin/env python3
"""Run all pipelines side by side and report RQ1 (temporal relevance), RQ2 (conflict resolution),
RQ3 (latency + token overhead). Writes results/summary.csv, results/per_query.csv, results/run_config.json.

Examples
  python evaluate.py --embedder tfidf                       # fully offline (rule reader/arbitrator)
  python evaluate.py --llm groq --arbitrator-mode hybrid --limit 40
  python evaluate.py --split dev --half-life 180            # tune baseline hyper-parameters on DEV only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os

from src.config import load_config
from src.llm import make_llm
from src.metrics import aggregate, fmt, paired_bootstrap, score_query
from src.runner import ALL_PIPELINES, build_system, load_notes, load_queries, run_measured
from src.schema import Query

logging.basicConfig(level=logging.WARNING)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--pipelines", default=",".join(ALL_PIPELINES))
    ap.add_argument("--limit", type=int, default=0, help="use only the first N queries (0 = all)")
    ap.add_argument("--llm", default=None, help="none|groq|gemini|ollama (default: LLM_PROVIDER)")
    ap.add_argument("--arbitrator-mode", default=None, help="rules|hybrid|llm (default: ARBITRATOR_MODE)")
    ap.add_argument("--embedder", default=None, help="minilm|tfidf (default: EMBEDDER)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--half-life", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = load_config()
    if a.llm: cfg.llm_provider = a.llm
    if a.arbitrator_mode: cfg.arbitrator_mode = a.arbitrator_mode
    if a.embedder: cfg.embedder = a.embedder
    if a.top_k: cfg.top_k = a.top_k
    if a.half_life: cfg.half_life_days = a.half_life
    if a.alpha is not None: cfg.alpha = a.alpha

    notes = load_notes(os.path.join(a.data, "notes.jsonl"))
    queries = load_queries(os.path.join(a.data, "queries.json"), a.split)
    if a.limit:
        queries = queries[: a.limit]
    llm = make_llm(cfg.llm_provider)
    pipelines, meter, report = build_system(cfg, notes, llm=llm)
    names = [p.strip() for p in a.pipelines.split(",") if p.strip()]

    print(f"Ingestion: {report}")
    print(f"Queries: {len(queries)} (split={a.split}) | reader={report['reader']} | "
          f"arbitrator={report['arbitrator_mode']} | embedder={report['embedder']} | top_k={cfg.top_k}")
    if report["reader"] == "rules":
        print("NOTE: no LLM configured -> accuracy uses the offline rule reader (a proxy). "
              "Report LLM-reader numbers for the final results.\n")

    rows = {n: [] for n in names}
    for i, g in enumerate(queries, 1):
        q = Query(g["query_id"], g["patient_id"], g["text"])
        for n in names:
            res = run_measured(pipelines[n], q, meter)
            rows[n].append(score_query(g, res))
        if i % 25 == 0:
            print(f"  ...{i}/{len(queries)} queries")

    summary = {n: aggregate(rows[n]) for n in names}
    cols = [("accuracy", True), ("clean_context", True), ("evidence_recall", True), ("stale_rate", True),
            ("exposure", True), ("conflict_resolution_rate", True), ("latency_ms", False),
            ("context_tokens", False), ("arbitration_tokens", False), ("llm_calls", False)]
    head = ["pipeline"] + ["acc", "clean_ctx", "ev_recall", "stale", "exposure", "CRR", "lat_ms", "ctx_tok", "arb_tok", "llm_calls"]
    print("\n" + " | ".join(f"{h:>10}" for h in head))
    for n in names:
        print(" | ".join([f"{n:>10}"] + [f"{fmt(summary[n][k], pct, 1 if not pct else 2):>10}" for k, pct in cols]))
    print("\nacc=answer accuracy | clean_ctx=correct note present, no wrong-time note | stale=share of wrong-time notes in context\n"
          "exposure=queries whose context holds BOTH correct and wrong-time notes | CRR=conflict resolution rate\n")

    print("Breakdown by query type (clean_context / accuracy):")
    for kind in ("current", "as_of"):
        line = [f"{kind:>8}"]
        for n in names:
            sub = [r for r in rows[n] if r["kind"] == kind]
            if sub:
                s = aggregate(sub)
                line.append(f"{n}: {fmt(s['clean_context'], True, 1)}/{fmt(s['accuracy'], True, 1)} (n={len(sub)})")
        print("  " + "  ".join(line))

    if "full" in names:
        print("\nPaired bootstrap, 95% CI of (full - baseline) per query (CI excluding 0 => difference unlikely due to chance):")
        for base in [n for n in names if n != "full"]:
            for metric in ("clean_context", "accuracy"):
                r = paired_bootstrap([x[metric] for x in rows["full"]], [x[metric] for x in rows[base]])
                if r:
                    print(f"  full - {base:<10} {metric:<14} {r[0]:+.3f}  [{r[1]:+.3f}, {r[2]:+.3f}]")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["pipeline"] + list(next(iter(summary.values())).keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for n in names:
            w.writerow({"pipeline": n, **summary[n]})
    with open(os.path.join(a.out, "per_query.csv"), "w", newline="", encoding="utf-8") as f:
        allrows = [r for n in names for r in rows[n]]
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    with open(os.path.join(a.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "config": {k: str(v) for k, v in vars(cfg).items()}, "ingestion": report,
                   "n_queries": len(queries)}, f, indent=2)
    print(f"\nWrote {a.out}/summary.csv, per_query.csv, run_config.json")


if __name__ == "__main__":
    main()
