import json
import os
import subprocess
import sys
from datetime import datetime

from src.ner_ingestion import DateExtractor, note_to_chunk, select_document_date
from src.schema import Note

EX = DateExtractor("DMY")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def doc_date(text, filed=None):
    c = note_to_chunk(Note("n", "p", text, filed), EX)
    return (c.doc_date.date().isoformat() if c.doc_date else None), c.date_source, c


def test_header_formats():
    for text, want in [("Clinic visit on 3 March 2021. x", "2021-03-03"), ("Visit date: 2020-11-19. x", "2020-11-19"),
                       ("Seen on October 5, 2022. x", "2022-10-05"), ("Date of consultation: 12/05/2023. x", "2023-05-12"),
                       ("Seen on 3rd March 2021. x", "2021-03-03")]:
        assert doc_date(text)[0] == want


def test_date_order_dmy():
    assert doc_date("Date of consultation: 03-04-2020. ok")[0] == "2020-04-03"     # 3 April, not 4 March


def test_relative_date_resolved_against_note_date():
    _, _, chunk = doc_date("Clinic visit on 27 September 2024. Previous review was 3 months ago.")
    assert "2024-06-27" in chunk.mentioned_dates


def test_duration_is_not_the_document_date():
    assert doc_date("Visit date: 2020-11-19. cough for one week.")[0] == "2020-11-19"


def test_fallback_and_missing():
    d, src, _ = doc_date("No date in this note body.", filed="2022-05-17")
    assert (d, src) == ("2022-05-17", "fallback")
    d, src, _ = doc_date("No date in this note body.", filed=None)
    assert (d, src) == (None, "missing")


def test_templates_match_lexicon():
    sys.path.insert(0, ROOT)
    from scripts.generate_dataset import validate_templates
    validate_templates()      # raises AssertionError on mismatch


def test_dataset_is_reproducible(tmp_path):
    outs = []
    for i in range(2):
        d = tmp_path / str(i)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "generate_dataset.py"), "--patients", "6",
                        "--seed", "7", "--out", str(d)], check=True, capture_output=True)
        outs.append(open(d / "notes.jsonl").read() + open(d / "queries.json").read())
    assert outs[0] == outs[1]
