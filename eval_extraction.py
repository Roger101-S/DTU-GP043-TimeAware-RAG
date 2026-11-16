#!/usr/bin/env python3
"""Measure date-extraction accuracy (supports the RQ1 claim that NER-derived metadata is reliable).

Two checks:
  1. data/extraction_gold.json : a small HAND-WRITTEN sample with varied date formats. Extend it with notes you
                                 label yourselves; the more independent labels, the more credible the number.
  2. synthetic dataset         : visit_date known from the generator (same templates as the pipeline sees).
Reports document-date accuracy overall and by header style / method, and lists every miss.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from src.config import load_config
from src.ner_ingestion import DateExtractor, note_to_chunk
from src.runner import load_notes
from src.schema import Note


def run_sample(extractor, items):
    hits, rows = 0, []
    src_hits = 0
    ment_tot = ment_hit = 0
    for it in items:
        c = note_to_chunk(Note(it["id"], "X", it["text"], it.get("filed_date")), extractor)
        got = c.doc_date.date().isoformat() if c.doc_date else None
        ok = got == it["expected_doc_date"]
        hits += ok
        src_hits += c.date_source == it["expected_source"]
        for m in it.get("expected_mentions", []):
            ment_tot += 1
            ment_hit += m in c.mentioned_dates
        rows.append((it["id"], ok, got, it["expected_doc_date"], c.date_source, it["text"][:70]))
    return hits, src_hits, ment_hit, ment_tot, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    cfg = load_config()
    ex = DateExtractor(cfg.date_order)
    print(f"Extractor: {ex.method_name} | DATE_ORDER={cfg.date_order}\n")

    items = json.load(open(os.path.join(a.data, "extraction_gold.json"), encoding="utf-8"))
    hits, src_hits, mh, mt, rows = run_sample(ex, items)
    print(f"[Hand-written sample] n={len(items)}  document-date accuracy: {hits}/{len(items)} ({100 * hits / len(items):.0f}%)  "
          f"source correct: {src_hits}/{len(items)}  other-date recall: {mh}/{mt}")
    for rid, ok, got, exp, src, txt in rows:
        if not ok:
            print(f"   MISS {rid}: got {got} ({src}), expected {exp} | {txt}")

    notes = load_notes(os.path.join(a.data, "notes.jsonl"))
    gold = json.load(open(os.path.join(a.data, "gold_annotations.json"), encoding="utf-8"))
    per_style, per_src = Counter(), Counter()
    tot_style = Counter()
    ok_total = 0
    for n in notes:
        c = note_to_chunk(n, ex)
        style = n.text.split(".")[0].split(":")[0].split(" on ")[0][:22]
        correct = c.doc_date is not None and c.doc_date.date().isoformat() == gold[n.note_id]["visit_date"]
        ok_total += correct
        tot_style[style] += 1
        per_style[style] += correct
        per_src[c.date_source] += 1
    print(f"\n[Synthetic notes] n={len(notes)}  document-date accuracy: {ok_total}/{len(notes)} ({100 * ok_total / len(notes):.1f}%)")
    print("  by header style:", {k: f"{per_style[k]}/{tot_style[k]}" for k in tot_style})
    print("  date source used:", dict(per_src), "(fallback = system filing date, lags the visit by 0-14 days)")


if __name__ == "__main__":
    main()
