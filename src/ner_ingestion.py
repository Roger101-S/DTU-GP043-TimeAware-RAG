"""Temporal metadata extraction (addresses G1 / RQ1).

Pipeline for every raw note:
    spaCy NER (label DATE)  +  regex supplement for numeric dates  ->  dateparser  ->  datetime
then pick the *document date* (the visit date, normally in the header) and attach it, together with
the other dates mentioned in the text, as HARD METADATA next to the embedding.

Design choices (explain these in the viva):
  * spaCy's small English model is not trained on clinical text and misses some numeric formats such as
    14/03/2022, so a small regex supplement catches those. The extraction method is recorded per date.
  * Relative expressions ("3 weeks ago") are resolved against the DOCUMENT date, not today's date.
  * DATE_ORDER (DMY by default) removes the 05/06 ambiguity.  Change it in .env if your data is MDY.
  * If no absolute date is found near the start of the note, we fall back to the system filing date
    (flagged date_source="fallback"); if that is missing too the note is excluded (date_source="missing").
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import dateparser

from .schema import Chunk, Note

log = logging.getLogger(__name__)

_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")
_RELATIVE = re.compile(r"\b(ago|yesterday|today|tomorrow|last|next|this|previous)\b", re.I)
_HAS_YEAR = re.compile(r"\b(19|20)\d{2}\b")

_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_HAS_MONTH = re.compile(rf"\b{_MONTHS}\b", re.I)

# FULL-date formats are matched by regex FIRST. Reason (measured on our hand-labelled sample): the small
# spaCy model labels ISO dates as CARDINAL, splits "3rd March 2021" into pieces and tags "9 February 2026"
# as EVENT. spaCy is kept for everything else (partial/relative expressions such as "June 2021", "3 weeks ago").
_FULL_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                                                    # 2022-03-14
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+{_MONTHS}\.?,?\s+\d{{4}}\b", re.I),   # 14 March 2022, 3rd March 2021
    re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I),          # March 14, 2022
    _NUMERIC_DATE,                                                                              # 14/03/2022
]
_MONTH_YEAR = re.compile(rf"\b{_MONTHS}\.?\s+\d{{4}}\b", re.I)                                # June 2021 (no-spaCy fallback)


@dataclass
class ExtractedDate:
    text: str
    start: int
    end: int
    value: Optional[datetime]
    relative: bool
    method: str            # "spacy" | "regex" | "regex-fallback"


class DateExtractor:
    """spaCy DATE entities (+ regex supplement) resolved to datetimes with dateparser."""

    def __init__(self, date_order: str = "DMY", use_spacy: bool = True, model: str = "en_core_web_sm"):
        self.date_order = date_order
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(model, disable=["lemmatizer"])
            except Exception as exc:  # model missing or spaCy not installed
                log.warning("spaCy model '%s' unavailable (%s). Falling back to regex-only date extraction. "
                            "Run: python -m spacy download %s", model, exc, model)

    @property
    def method_name(self) -> str:
        return "spacy+dateparser" if self.nlp is not None else "regex+dateparser (spaCy model missing)"

    def _settings(self, anchor: Optional[datetime], prefer_day: str) -> dict:
        s = {
            "DATE_ORDER": self.date_order,
            "PREFER_DATES_FROM": "past",
            "PREFER_DAY_OF_MONTH": prefer_day,
            "RETURN_AS_TIMEZONE_AWARE": False,
        }
        if anchor is not None:
            s["RELATIVE_BASE"] = anchor
        return s

    def _spans(self, text: str) -> list:
        spans = []

        def free(start, end):
            return not any(st < end and start < en for st, en, _, _ in spans)

        for pat in _FULL_DATE_PATTERNS:                      # 1) full dates by regex (most specific)
            for m in pat.finditer(text):
                if free(m.start(), m.end()):
                    spans.append((m.start(), m.end(), m.group(0), "regex"))
        if self.nlp is not None:                             # 2) spaCy DATE entities for everything else
            for e in self.nlp(text).ents:
                if e.label_ == "DATE" and free(e.start_char, e.end_char):
                    spans.append((e.start_char, e.end_char, e.text, "spacy"))
        else:                                                # 2') no model: month-year by regex
            for m in _MONTH_YEAR.finditer(text):
                if free(m.start(), m.end()):
                    spans.append((m.start(), m.end(), m.group(0), "regex-fallback"))
        return sorted(spans)

    def extract(self, text: str, anchor: Optional[datetime] = None, prefer_day: str = "first") -> list:
        out = []
        for start, end, span_text, method in self._spans(text):
            has_year = bool(_HAS_YEAR.search(span_text))
            has_month = bool(_HAS_MONTH.search(span_text))
            # durations/relative phrases ("one week", "3 weeks ago", "last month") never carry a year or month name
            relative = (not has_year and not has_month) or (bool(_RELATIVE.search(span_text)) and not has_year)
            if re.fullmatch(r"(19|20)\d{2}", span_text.strip()):
                # bare year: dateparser would borrow today's month/day, so pin it explicitly
                y = int(span_text.strip())
                value = datetime(y, 12, 31) if prefer_day == "last" else datetime(y, 1, 1)
            else:
                # relative phrases keep the anchor's day of month ("3 months ago"); absolute partial dates use prefer_day
                value = dateparser.parse(span_text, settings=self._settings(anchor, "current" if relative else prefer_day))
            out.append(ExtractedDate(span_text, start, end, value, relative, method))
        return out


def select_document_date(extracted: list, fallback_iso: Optional[str], header_chars: int = 160):
    """Pick the visit date: first absolute, parseable date near the top of the note."""
    for d in extracted:
        if d.value is not None and not d.relative and d.start < header_chars:
            return d.value, "text"
    if fallback_iso:
        try:
            return datetime.fromisoformat(fallback_iso), "fallback"
        except ValueError:
            log.warning("Unparseable fallback date %r", fallback_iso)
    return None, "missing"


def note_to_chunk(note: Note, extractor: DateExtractor) -> Chunk:
    extracted = extractor.extract(note.text)
    doc_date, source = select_document_date(extracted, note.filed_date)
    mentioned = []
    for d in extracted:
        value = d.value
        if d.relative and doc_date is not None:               # resolve "3 weeks ago" against the visit date
            import dateparser as _dp
            value = _dp.parse(d.text, settings=extractor._settings(doc_date, "current"))
        if value is not None and value != doc_date:
            mentioned.append(value.date().isoformat())
    return Chunk(note_id=note.note_id, patient_id=note.patient_id, text=note.text,
                 doc_date=doc_date, date_source=source, mentioned_dates=sorted(set(mentioned)))


def build_chunks(notes: list, extractor: DateExtractor):
    """Run extraction on every note. Returns (chunks_with_dates, ingestion_report)."""
    chunks, report = [], {"text": 0, "fallback": 0, "missing": 0, "total": len(notes),
                          "extractor": extractor.method_name}
    for n in notes:
        c = note_to_chunk(n, extractor)
        report[c.date_source] += 1
        if c.doc_date is None:
            log.warning("Note %s has no usable date and was excluded.", n.note_id)
            continue
        chunks.append(c)
    return chunks, report
