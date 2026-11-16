#!/usr/bin/env python3
"""Show one query through every pipeline: context, answer, and the arbitrator's audit log. Good for the viva demo.

  python scripts/demo.py --query-id Q0003 --embedder tfidf
  python scripts/demo.py --text "What is the current hypertension status of patient P002?" --patient P002
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config  # noqa: E402
from src.llm import make_llm  # noqa: E402
from src.runner import ALL_PIPELINES, build_system, load_notes, load_queries, run_measured  # noqa: E402
from src.schema import Query  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
ap.add_argument("--query-id")
ap.add_argument("--text")
ap.add_argument("--patient")
ap.add_argument("--embedder")
ap.add_argument("--llm")
ap.add_argument("--arbitrator-mode")
a = ap.parse_args()

cfg = load_config()
if a.embedder: cfg.embedder = a.embedder
if a.llm: cfg.llm_provider = a.llm
if a.arbitrator_mode: cfg.arbitrator_mode = a.arbitrator_mode
notes = load_notes(os.path.join(a.data, "notes.jsonl"))
gold = None
if a.query_id:
    gold = next(q for q in load_queries(os.path.join(a.data, "queries.json")) if q["query_id"] == a.query_id)
    q = Query(gold["query_id"], gold["patient_id"], gold["text"])
else:
    q = Query("adhoc", a.patient, a.text)

pipelines, meter, report = build_system(cfg, notes, llm=make_llm(cfg.llm_provider))
print(f"QUERY: {q.text}  (patient {q.patient_id})")
if gold:
    print(f"GOLD : state={gold['gold_state']} | correct note={gold['gold_note_id']} | "
          f"superseded={gold['superseded_note_ids']} | future={gold['future_note_ids']}")
by_id = {n.note_id: n for n in notes}
for name in ALL_PIPELINES:
    r = run_measured(pipelines[name], q, meter)
    print(f"\n=== {name} ===  answer: {r.answer_state}  | {r.latency_ms:.1f} ms | ctx tokens ~{r.context_tokens} | llm calls {r.llm_calls}")
    for c in sorted(r.context, key=lambda c: c.doc_date):
        print(f"   [{c.note_id}] {c.doc_date.date()}  {by_id[c.note_id].text[:110]}")
    for s in r.suppressed:
        print(f"   SUPPRESSED {s['note_id']} ({s['date']}, {s['old_state']}) <- {s['superseded_by']} ({s['new_date']}, {s['new_state']}) reason={s['reason']} via {s['method']}")
