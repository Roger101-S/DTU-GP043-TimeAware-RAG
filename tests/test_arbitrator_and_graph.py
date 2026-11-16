from datetime import datetime

from tests.conftest import FakeLLM
from src.arbitrator import Arbitrator, resolve_supersession, Observation
from src.llm import TokenMeter
from src.schema import Chunk, Query


def chunk(i, d, text):
    return Chunk(i, "P", text, datetime.fromisoformat(d), "text")


C1 = chunk("n1", "2020-03-10", "Clinic visit on 10 March 2020. No history of hypertension.")
C2 = chunk("n2", "2022-06-15", "Visit date: 2022-06-15. Pre-hypertension noted, BP 132/86.")
C3 = chunk("n3", "2024-01-20", "Seen on January 20, 2024. Stage 2 hypertension, BP 168/104.")
C4 = chunk("n4", "2024-03-01", "Seen on 1 March 2024. BP remains markedly elevated at 170/105 despite therapy.")  # paraphrase
C5 = chunk("n5", "2023-05-05", "Routine visit; vaccinations up to date.")


def test_rules_suppress_superseded_with_audit_log():
    d = Arbitrator("rules").arbitrate([C1, C2, C3, C5], "hypertension")
    assert d.current_state == "stage2_hypertension" and d.conflict_detected
    assert {x["note_id"] for x in d.suppressed} == {"n1", "n2"}
    assert {c.note_id for c in d.kept} == {"n3", "n5"}               # unrelated routine note is untouched
    entry = [x for x in d.suppressed if x["note_id"] == "n1"][0]
    assert entry["superseded_by"] == "n3" and entry["old_state"] == "none" and entry["reason"] == "superseded"


def test_same_state_notes_are_corroborating_not_suppressed():
    c_dup = chunk("n3b", "2024-02-02", "Visit date: 2024-02-02. Known stage 2 hypertension; BP 166/100 on review.")
    d = Arbitrator("rules").arbitrate([C1, C3, c_dup], "hypertension")
    assert {c.note_id for c in d.kept} == {"n3", "n3b"}


def test_abstains_when_newer_unclassified_note_exists():
    d = Arbitrator("rules").arbitrate([C1, C3, C4], "hypertension")
    assert d.abstained and d.suppressed == [] and len(d.kept) == 3


def test_older_unclassified_dropped_with_reason():
    old_para = chunk("n0", "2019-01-01", "Visit date: 2019-01-01. Blood pressure unremarkable at 118/76.")
    d = Arbitrator("rules").arbitrate([old_para, C3], "hypertension")
    assert d.suppressed[0]["reason"] == "older_unclassified"


def test_same_day_conflict_abstains():
    a = chunk("x1", "2024-01-20", "Stage 1 hypertension noted.")
    b = chunk("x2", "2024-01-20", "Stage 2 hypertension noted.")
    d = Arbitrator("rules").arbitrate([a, b], "hypertension")
    assert d.abstained and not d.suppressed


def test_unknown_condition_is_passthrough():
    d = Arbitrator("rules").arbitrate([C1, C3], None)
    assert len(d.kept) == 2 and not d.suppressed


def test_hybrid_uses_llm_only_for_unclassified_notes():
    llm = FakeLLM('{"n4": "stage2_hypertension"}')
    meter = TokenMeter()
    d = Arbitrator("hybrid", llm=llm, meter=meter).arbitrate([C1, C3, C4], "hypertension")
    assert meter.calls == 1 and "n1" not in llm.prompts[0] and "n4" in llm.prompts[0]
    assert d.current_state == "stage2_hypertension"
    assert {x["note_id"] for x in d.suppressed} == {"n1"}               # n3 has the same state as n4: corroborates
    assert {c.note_id for c in d.kept} == {"n3", "n4"}
    assert [x for x in d.suppressed][0]["method"] == "rules"


def test_llm_mode_parses_json_with_fences_and_ignores_invalid_labels():
    reply = '```json\n{"n1": "none", "n3": "stage2_hypertension", "n5": "banana"}\n```'
    d = Arbitrator("llm", llm=FakeLLM(reply), meter=TokenMeter()).arbitrate([C1, C3, C5], "hypertension")
    assert {x["note_id"] for x in d.suppressed} == {"n1"} and "n5" in {c.note_id for c in d.kept}
    assert d.suppressed[0]["method"] == "llm"


def test_llm_mode_without_llm_degrades_to_rules():
    assert Arbitrator("llm", llm=None).mode == "rules"


def test_full_pipeline_end_to_end(system):
    pipelines, meter, report, _ = system
    q = Query("q", "PA", "What is the current hypertension status of patient PA?")
    res = pipelines["full"].run(q)
    assert res.answer_state == "stage2_hypertension"
    assert {c.note_id for c in res.context} == {"A-3", "A-4"}
    assert {s["note_id"] for s in res.suppressed} == {"A-1", "A-2"}
    assert pipelines["naive"].run(q).answer_state == "CONFLICT"      # naive RAG sees contradictory evidence


def test_as_of_query_full_pipeline(system):
    pipelines, *_ = system
    res = pipelines["full"].run(Query("q", "PA", "What was the hypertension status of patient PA as of July 2022?"))
    assert res.answer_state == "prehypertension" and "A-3" not in {c.note_id for c in res.context}


def test_router_skips_arbitrator_when_no_conflict_possible(system):
    pipelines, *_ = system
    res = pipelines["full"].run(Query("q", "PB", "What is the current hypertension status of patient PB?"))
    assert res.arbitrated is False
