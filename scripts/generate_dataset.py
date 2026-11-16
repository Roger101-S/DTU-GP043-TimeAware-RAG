#!/usr/bin/env python3
"""Generate a SYNTHETIC longitudinal clinical-notes dataset with ground-truth labels.

Everything here is invented text. It contains no real patient data. It is meant to test the temporal
pipeline, not to claim clinical realism. Real evidence for the report needs MIMIC-style data (see README).

Outputs (in --out, default ./data):
  notes.jsonl           raw notes the pipelines see: note_id, patient_id, text, filed_date
  queries.json          queries + GOLD labels (gold labels are read only by the metrics code)
  gold_annotations.json per-note truth: visit_date + condition/state annotations (for extraction eval)

Design points:
  * Each patient has several visits. For 2-3 conditions the stated status EVOLVES over time
    (e.g. none -> pre-hypertension -> stage 2), so older notes are genuinely superseded.
  * ~30% of mentions use PARAPHRASED wording the rule lexicon does not recognise (hard cases).
  * Visit dates appear in the text in several formats; `filed_date` (system date) lags the visit by 0-14 days
    and is used only as an extraction fallback.
  * Queries are "current status" (answered as of 2026-01-01) or "as of <Month Year>" / "<D Month Year>".
  * Patients are split 30% dev / 70% test (by patient) so you can tune on dev and report on test.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import random
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.medical_lexicon import CONDITIONS, classify_sentence, condition_states  # noqa: E402

REFERENCE = date(2026, 1, 1)

T = {  # condition -> state -> {"canon": lexicon-recognised, "para": NOT recognised by the lexicon}
    "hypertension": {
        "none": {"canon": ["No history of hypertension.", "Hypertension not present; BP {sbp}/{dbp} within normal limits."],
                 "para": ["Normotensive on all readings today, BP {sbp}/{dbp}.", "Blood pressure unremarkable at {sbp}/{dbp}."]},
        "prehypertension": {"canon": ["Pre-hypertension noted, BP {sbp}/{dbp}.", "Findings consistent with prehypertension; BP {sbp}/{dbp}."],
                            "para": ["BP borderline at {sbp}/{dbp}, lifestyle advice given.", "Readings creeping upward ({sbp}/{dbp}); recheck advised."]},
        "stage1_hypertension": {"canon": ["Diagnosed with stage 1 hypertension, BP {sbp}/{dbp}.", "Stage 1 hypertension; lifestyle modification advised, BP {sbp}/{dbp}."],
                                "para": ["Persistently raised BP {sbp}/{dbp} on repeat checks; antihypertensive started.", "Home BP log averaging {sbp}/{dbp}; started on low-dose ACE inhibitor."]},
        "stage2_hypertension": {"canon": ["Stage 2 hypertension, BP {sbp}/{dbp}, started on amlodipine 5 mg.", "Known stage 2 hypertension; BP {sbp}/{dbp} on review."],
                                "para": ["BP remains markedly elevated at {sbp}/{dbp} despite therapy.", "Uncontrolled readings ({sbp}/{dbp}); second agent added."]},
    },
    "type2_diabetes": {
        "none": {"canon": ["No history of diabetes.", "Diabetes not present; fasting glucose {glu} mg/dL."],
                 "para": ["Fasting glucose {glu} mg/dL, normal.", "Glucose within normal range ({glu} mg/dL)."]},
        "prediabetes": {"canon": ["Pre-diabetes noted, fasting glucose {glu} mg/dL.", "Prediabetes; HbA1c {a1c}%."],
                        "para": ["Fasting glucose borderline at {glu} mg/dL; diet review advised.", "HbA1c {a1c}%, in the intermediate range."]},
        "type2_diabetes": {"canon": ["Diagnosed with type 2 diabetes, HbA1c {a1c}%.", "Type 2 diabetes mellitus; started metformin 500 mg, HbA1c {a1c}%."],
                           "para": ["HbA1c {a1c}% on repeat testing, metformin started.", "Glycaemic control suboptimal (HbA1c {a1c}%); dose of metformin increased."]},
    },
    "asthma": {
        "none": {"canon": ["No history of asthma.", "No diagnosis of asthma; chest clear on auscultation."],
                 "para": ["Chest clear, no wheeze, no respiratory complaints.", "No cough or breathlessness reported."]},
        "mild_intermittent_asthma": {"canon": ["Mild intermittent asthma, salbutamol as needed.", "Diagnosed with mild intermittent asthma."],
                                     "para": ["Occasional wheeze on exertion; reliever inhaler as needed.", "Intermittent breathlessness, responds to salbutamol."]},
        "moderate_persistent_asthma": {"canon": ["Moderate persistent asthma; inhaled corticosteroid started.", "Moderate persistent asthma, peak flow {pf}% of predicted."],
                                       "para": ["Daily symptoms with night waking; preventer inhaler started.", "Frequent exacerbations despite reliever use; step-up therapy planned."]},
    },
    "anemia": {
        "none": {"canon": ["No history of anemia.", "No evidence of anemia; hemoglobin {hb} g/dL."],
                 "para": ["Haemoglobin {hb} g/dL, within range.", "Blood count unremarkable (Hb {hb} g/dL)."]},
        "iron_deficiency_anemia": {"canon": ["Iron-deficiency anemia, hemoglobin {hb} g/dL; iron supplementation started.", "Diagnosed with iron deficiency anemia, Hb {hb} g/dL."],
                                   "para": ["Low hemoglobin ({hb} g/dL) with low ferritin; iron started.", "Hb {hb} g/dL, microcytic picture, ferritin low."]},
        "resolved": {"canon": ["Iron-deficiency anemia resolved; hemoglobin {hb} g/dL.", "Anemia has resolved, hemoglobin {hb} g/dL."],
                     "para": ["Hemoglobin normalised at {hb} g/dL; iron stopped.", "Blood counts back to normal (Hb {hb} g/dL), supplementation discontinued."]},
    },
    "hypothyroidism": {
        "none": {"canon": ["No history of thyroid disease.", "No evidence of hypothyroidism; TSH {tsh} mIU/L."],
                 "para": ["TSH {tsh} mIU/L, normal.", "Thyroid function within normal limits."]},
        "subclinical_hypothyroidism": {"canon": ["Subclinical hypothyroidism, TSH {tsh} mIU/L.", "Findings consistent with subclinical hypothyroidism; TSH {tsh} mIU/L."],
                                       "para": ["TSH mildly raised at {tsh} mIU/L with normal free T4; monitoring.", "Borderline TSH {tsh} mIU/L, repeat in 3 months."]},
        "hypothyroidism": {"canon": ["Hypothyroidism, TSH {tsh} mIU/L; levothyroxine started.", "Established hypothyroidism on levothyroxine, TSH {tsh} mIU/L."],
                           "para": ["TSH {tsh} mIU/L, levothyroxine dose adjusted.", "Persistently raised TSH ({tsh} mIU/L); replacement therapy commenced."]},
    },
}

ROUTINE = ["Routine visit; vaccinations up to date.", "Seen for seasonal cold, symptomatic treatment advised.",
           "Annual review: weight stable, no new complaints.", "Dressing change for minor wound, healing well.",
           "Follow-up of minor ankle sprain, improving."]


def vals(cond, state, rng):
    r = rng.randint
    sbp_dbp = {"none": ((108, 124), (68, 80)), "prehypertension": ((126, 138), (82, 88)),
               "stage1_hypertension": ((140, 158), (90, 98)), "stage2_hypertension": ((160, 185), (100, 115))}
    v = {}
    if cond == "hypertension":
        (a, b), (c, d) = sbp_dbp[state]
        v["sbp"], v["dbp"] = r(a, b), r(c, d)
    if cond == "type2_diabetes":
        g = {"none": (82, 98), "prediabetes": (101, 124), "type2_diabetes": (130, 190)}[state]
        h = {"none": (50, 56), "prediabetes": (57, 64), "type2_diabetes": (65, 95)}[state]
        v["glu"], v["a1c"] = r(*g), r(*h) / 10
    if cond == "asthma":
        v["pf"] = {"none": r(90, 105), "mild_intermittent_asthma": r(80, 90), "moderate_persistent_asthma": r(55, 70)}[state]
    if cond == "anemia":
        lo, hi = {"none": (125, 150), "iron_deficiency_anemia": (75, 105), "resolved": (122, 145)}[state]
        v["hb"] = r(lo, hi) / 10
    if cond == "hypothyroidism":
        lo, hi = {"none": (10, 35), "subclinical_hypothyroidism": (46, 95), "hypothyroidism": (100, 400)}[state]
        v["tsh"] = r(lo, hi) / 10
    return v


def fmt_long(d):  return f"{d.day} {d.strftime('%B %Y')}"
def fmt_us(d):    return f"{d.strftime('%B')} {d.day}, {d.year}"
HEADERS = [
    lambda d: f"Clinic visit on {fmt_long(d)}.",
    lambda d: f"Visit date: {d.isoformat()}.",
    lambda d: f"Seen on {fmt_us(d)}.",
    lambda d: f"Date of consultation: {d.day:02d}/{d.month:02d}/{d.year}.",
]


def month_end(d):
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def sample_visit_dates(rng, n):
    lo, hi = date(2018, 1, 1), date(2025, 10, 31)
    span = (hi - lo).days
    for _ in range(500):
        offs = sorted(rng.sample(range(span), n))
        if all(b - a >= 35 for a, b in zip(offs, offs[1:])):
            return [lo + timedelta(days=o) for o in offs]
    raise RuntimeError("could not sample visit dates")


def make_path(cond, m, rng):
    states = condition_states(cond)
    L = min(rng.randint(2, min(len(states), 4)), m)
    s0 = rng.randint(0, len(states) - L)
    path = states[s0:s0 + L]
    counts = [1] * L
    for _ in range(m - L):
        counts[rng.randrange(L)] += 1
    return [s for s, c in zip(path, counts) for _ in range(c)]


def generate(n_patients, seed, p_para):
    rng = random.Random(seed)
    notes, gold_notes, queries = [], {}, []
    mention_log = {}   # (pid, cond) -> list of (note_id, date, state)
    for pi in range(n_patients):
        pid = f"P{pi + 1:03d}"
        split = "dev" if pi % 10 < 3 else "test"
        age, sex = rng.randint(25, 85), rng.choice(["male", "female"])
        n_visits = rng.randint(10, 16)
        dates = sample_visit_dates(rng, n_visits)
        conds = rng.sample(list(T), k=rng.choice([2, 3]))
        per_visit = {i: [] for i in range(n_visits)}
        for cond in conds:
            m = rng.randint(3, 5)
            idxs = sorted(rng.sample(range(n_visits), m))
            for i, st in zip(idxs, make_path(cond, m, rng)):
                per_visit[i].append((cond, st))
        for i, d in enumerate(dates):
            note_id = f"{pid}-N{i + 1:02d}"
            parts = [rng.choice(HEADERS)(d), f"Patient {pid}, {age}-year-old {sex}."]
            ann = []
            for cond, st in per_visit[i]:
                para = rng.random() < p_para
                tmpl = rng.choice(T[cond][st]["para" if para else "canon"])
                parts.append(tmpl.format(**vals(cond, st, rng)))
                ann.append({"condition": cond, "state": st, "paraphrased": para})
                mention_log.setdefault((pid, cond), []).append((note_id, d, st))
            if not ann:
                parts.append(rng.choice(ROUTINE))
            if rng.random() < 0.2:
                parts.append(f"Vaccination record updated on {fmt_long(date(rng.randint(2015, 2017), rng.randint(1, 12), rng.randint(1, 28)))}.")
            if rng.random() < 0.2:
                parts.append(f"Previous review was {rng.randint(2, 9)} months ago.")
            notes.append({"note_id": note_id, "patient_id": pid, "text": " ".join(parts),
                          "filed_date": (d + timedelta(days=rng.randint(0, 14))).isoformat()})
            gold_notes[note_id] = {"visit_date": d.isoformat(), "annotations": ann}
        # queries
        for cond in conds:
            ms = mention_log[(pid, cond)]
            disp = CONDITIONS[cond]["display"]
            queries.append(_query(len(queries), pid, cond, split, ms, len(ms) - 1, None,
                                  rng.choice([f"What is the current {disp} status of patient {pid}?",
                                              f"What is the latest recorded {disp} status for patient {pid}?"]),
                                  REFERENCE, "current"))
            order = list(range(len(ms) - 1))
            rng.shuffle(order)
            for j in order:
                nxt = ms[j + 1][1]
                me = month_end(ms[j][1])
                if me < nxt:
                    txt = f"What was the {disp} status of patient {pid} as of {me.strftime('%B %Y')}?"
                    queries.append(_query(len(queries), pid, cond, split, ms, j, me, txt, me, "as_of"))
                    break
                dd = ms[j][1] + timedelta(days=10)
                if dd < nxt:
                    txt = f"What was the {disp} status of patient {pid} as of {fmt_long(dd)}?"
                    queries.append(_query(len(queries), pid, cond, split, ms, j, dd, txt, dd, "as_of"))
                    break
    return notes, gold_notes, queries


def _query(n, pid, cond, split, ms, j, as_of, text, as_of_date, kind):
    gold_id, gold_date, gold_state = ms[j][0], ms[j][1], ms[j][2]
    cutoff = as_of_date
    visible = [x for x in ms if x[1] <= cutoff]
    return {
        "query_id": f"Q{n + 1:04d}", "patient_id": pid, "text": text, "split": split, "kind": kind,
        "condition": cond, "as_of": as_of_date.isoformat(),
        "gold_state": gold_state, "gold_note_id": gold_id,
        "superseded_note_ids": [x[0] for x in visible if x[2] != gold_state and x[1] < gold_date],
        "future_note_ids": [x[0] for x in ms if x[1] > cutoff],
    }


def validate_templates():
    """Canonical wording must be recognised by the lexicon; paraphrases must NOT be classified."""
    for cond, states in T.items():
        for st, d in states.items():
            fill = {"sbp": 150, "dbp": 95, "glu": 110, "a1c": 6.0, "pf": 80, "hb": 12.0, "tsh": 5.0}
            for s in d["canon"]:
                assert classify_sentence(s.format(**fill), cond) == st, f"canon mismatch: {cond}/{st}: {s}"
            for s in d["para"]:
                assert classify_sentence(s.format(**fill), cond) is None, f"para recognised: {cond}/{st}: {s}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paraphrase-prob", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    a = ap.parse_args()
    validate_templates()
    notes, gold, queries = generate(a.patients, a.seed, a.paraphrase_prob)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "notes.jsonl"), "w", encoding="utf-8") as f:
        for n in notes:
            f.write(json.dumps(n) + "\n")
    with open(os.path.join(a.out, "gold_annotations.json"), "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=1)
    with open(os.path.join(a.out, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=1)
    n_mentions = sum(len(v["annotations"]) for v in gold.values())
    n_para = sum(x["paraphrased"] for v in gold.values() for x in v["annotations"])
    print(f"patients={a.patients} notes={len(notes)} queries={len(queries)} "
          f"(dev={sum(q['split'] == 'dev' for q in queries)}, test={sum(q['split'] == 'test' for q in queries)}, "
          f"as_of={sum(q['kind'] == 'as_of' for q in queries)}) "
          f"condition-mentions={n_mentions} paraphrased={n_para} ({100 * n_para / max(1, n_mentions):.0f}%)")
