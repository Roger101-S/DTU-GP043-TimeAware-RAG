"""Query-aware temporal filtering (addresses G1 / RQ1).

The query is parsed with the SAME spaCy+dateparser extractor used at ingestion:
  * "as of June 2021"  -> evidence must have doc_date <= 30 June 2021   (hard bound pushed into the DB query)
  * "current/now/..."  -> evidence must have doc_date <= REFERENCE_DATE
Evidence that falls after the as-of date can therefore never reach the LLM, however similar it is.

Honest limitation: a date bound can remove FUTURE evidence but cannot, on its own, tell that an older
note has been superseded by a newer one. That is the job of the arbitration agent (arbitrator.py).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .medical_lexicon import detect_condition
from .ner_ingestion import DateExtractor
from .schema import Chunk, ParsedQuery


def parse_query(text: str, extractor: DateExtractor, reference_date: datetime) -> ParsedQuery:
    """Extract (condition, as-of date) from a natural-language query."""
    condition = detect_condition(text)
    # prefer_day="last": "as of June 2021" should include every note written in June 2021
    dates = [d for d in extractor.extract(text, anchor=reference_date, prefer_day="last")
             if d.value is not None and not d.relative]
    if dates:
        return ParsedQuery(condition=condition, as_of=dates[0].value, as_of_source="explicit")
    return ParsedQuery(condition=condition, as_of=reference_date, as_of_source="reference")


def date_to_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def temporal_bounds(parsed: ParsedQuery) -> tuple:
    """(min_ts, max_ts) for the metadata filter. Only an upper bound is needed for as-of semantics."""
    return None, date_to_ts(parsed.as_of)


def chronological_sort(chunks: list) -> list:
    """Oldest -> newest, so the generator reads the record as a timeline."""
    return sorted(chunks, key=lambda c: (c.doc_date, c.note_id))
