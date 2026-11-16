"""Plain data containers shared by every module."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Note:
    """A raw clinical note as it arrives from the EHR export. Only free text + a system filing date."""
    note_id: str
    patient_id: str
    text: str
    filed_date: Optional[str] = None   # ISO date the note was filed; used ONLY as an extraction fallback


@dataclass
class Chunk:
    """A note after temporal metadata extraction (one note = one chunk; notes are short)."""
    note_id: str
    patient_id: str
    text: str
    doc_date: Optional[datetime]
    date_source: str                   # "text" | "fallback" | "missing"
    mentioned_dates: list = field(default_factory=list)  # ISO strings of other dates found in the text
    score: float = 0.0                 # retrieval score (cosine similarity or blended score)


@dataclass
class Query:
    query_id: str
    patient_id: str
    text: str


@dataclass
class ParsedQuery:
    condition: Optional[str]           # canonical condition key, or None if not recognised
    as_of: datetime                    # evidence must be valid at this date
    as_of_source: str                  # "explicit" (date found in the query) | "reference" (current status)


@dataclass
class PipelineResult:
    query_id: str
    pipeline: str
    retrieved: list                    # chunks returned by retrieval, before any arbitration
    context: list                      # chunks actually passed to the generator
    answer: str
    answer_state: Optional[str]        # normalised state label the answer commits to
    suppressed: list                   # audit log: one dict per chunk removed by the arbitrator
    latency_ms: float
    context_tokens: int                # approximate tokens sent to the generator
    arbitration_tokens: int            # tokens spent by LLM calls inside the arbitrator (0 for rules)
    llm_calls: int                     # total LLM calls (arbitrator + generator)
    arbitrated: bool = False
    current_state: Optional[str] = None
