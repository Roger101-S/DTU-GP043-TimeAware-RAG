from datetime import datetime

from src.schema import Query
from src.temporal_filter import parse_query
from src.ner_ingestion import DateExtractor

EX = DateExtractor("DMY")
REF = datetime(2026, 1, 1)


def test_parse_current_query():
    p = parse_query("What is the current hypertension status of patient PA?", EX, REF)
    assert p.condition == "hypertension" and p.as_of == REF and p.as_of_source == "reference"


def test_parse_as_of_query_uses_end_of_month():
    p = parse_query("What was the hypertension status of patient PA as of June 2022?", EX, REF)
    assert p.as_of == datetime(2022, 6, 30) and p.as_of_source == "explicit"


def test_hard_filter_removes_future_notes(system):
    pipelines, *_ = system
    q = Query("q", "PA", "What was the hypertension status of patient PA as of July 2022?")
    res = pipelines["filter"].run(q)
    ids = {c.note_id for c in res.retrieved}
    assert "A-3" not in ids and {"A-1", "A-2"} <= ids          # the 2024 note can never be retrieved
    assert [c.note_id for c in res.retrieved] == sorted(ids, key=lambda i: i)  # chronological order


def test_patient_scoping(system):
    pipelines, *_ = system
    res = pipelines["naive"].run(Query("q", "PB", "What is the current hypertension status of patient PB?"))
    assert all(c.patient_id == "PB" for c in res.retrieved)


def test_naive_is_time_blind_and_decay_prefers_recent(system):
    pipelines, *_ = system
    q = Query("q", "PA", "What is the current hypertension status of patient PA?")
    naive = pipelines["naive"].run(q)
    assert {"A-1", "A-3"} <= {c.note_id for c in naive.context}        # stale 2020 note reaches the generator
    decay = pipelines["decay"].run(q)
    # soft decay is only a score: with alpha=0.6 similarity can still outrank recency, and the stale note leaks in
    assert "A-1" in {c.note_id for c in decay.context}


def test_decay_math_is_exponential_half_life(system):
    import copy
    pipelines, *_ = system
    pure_decay = copy.copy(pipelines["decay"])
    pure_decay.alpha = 0.0                                              # ignore similarity, keep only the time term
    res = pure_decay.run(Query("q", "PA", "What is the current hypertension status of patient PA?"))
    scores = {c.note_id: c.score for c in res.retrieved}
    age = (datetime(2026, 1, 1) - datetime(2024, 1, 20)).days
    assert abs(scores["A-3"] - 0.5 ** (age / 365.0)) < 1e-9             # alpha=0 -> score is exactly the decay term
    assert [c.note_id for c in res.retrieved] == ["A-3", "A-4", "A-2", "A-1"]   # newest first
