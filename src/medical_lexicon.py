"""Condition / state lexicon used by (a) query parsing, (b) the rule-based state extractor.

Each condition has:
  keywords : regexes that signal a sentence is *about* the condition
  states   : ORDERED list of (state_label, [regexes]); the first state whose regex matches wins,
             so negations ("no history of ...") and specific forms ("stage 2") are listed first.

This is deliberately small and transparent: it is the rule-based baseline arbitrator, and the
reason the LLM / hybrid mode exists is that wording outside this lexicon is not understood.
"""
from __future__ import annotations

import re

CONDITIONS = {
    "hypertension": {
        "display": "hypertension",
        "query_names": [r"hypertens", r"blood pressure"],
        "keywords": [r"hypertens", r"blood pressure", r"\bbp\b"],
        "states": [
            ("none", [r"no (known )?(history|diagnosis|evidence) of (hypertension|high blood pressure)",
                      r"hypertension (is )?(not present|absent|ruled out)",
                      r"denies (any )?(history of )?hypertension"]),
            ("stage2_hypertension", [r"stage (2|ii|two) hypertension"]),
            ("stage1_hypertension", [r"stage (1|i|one) hypertension"]),
            ("prehypertension", [r"pre-?hypertension"]),
        ],
    },
    "type2_diabetes": {
        "display": "type 2 diabetes",
        "query_names": [r"diabet"],
        "keywords": [r"diabet", r"glucose", r"hba1c"],
        "states": [
            ("none", [r"no (known )?(history|diagnosis|evidence) of (type 2 )?diabetes",
                      r"diabetes (is )?(not present|absent|ruled out)"]),
            ("prediabetes", [r"pre-?diabetes"]),
            ("type2_diabetes", [r"type 2 diabetes"]),
        ],
    },
    "asthma": {
        "display": "asthma",
        "query_names": [r"asthma"],
        "keywords": [r"asthma"],
        "states": [
            ("none", [r"no (known )?(history|diagnosis|evidence) of asthma"]),
            ("mild_intermittent_asthma", [r"mild intermittent asthma"]),
            ("moderate_persistent_asthma", [r"moderate persistent asthma"]),
        ],
    },
    "anemia": {
        "display": "anemia",
        "query_names": [r"an(a)?emi"],
        "keywords": [r"an(a)?emi", r"h(a)?emoglobin"],
        "states": [
            ("none", [r"no (known )?(history|diagnosis|evidence) of an(a)?emia"]),
            ("resolved", [r"an(a)?emia (has )?resolved",
                          r"resolved iron[- ]deficiency an(a)?emia",
                          r"iron[- ]deficiency an(a)?emia,? resolved"]),
            ("iron_deficiency_anemia", [r"iron[- ]deficiency an(a)?emia"]),
        ],
    },
    "hypothyroidism": {
        "display": "hypothyroidism",
        "query_names": [r"thyroid"],
        "keywords": [r"thyroid", r"\btsh\b"],
        "states": [
            ("none", [r"no (known )?(history|diagnosis|evidence) of (hypothyroidism|thyroid)"]),
            ("subclinical_hypothyroidism", [r"subclinical hypothyroidism"]),
            ("hypothyroidism", [r"hypothyroidism"]),
        ],
    },
}


def condition_states(condition: str) -> list:
    """Ordered state labels for a condition (used in LLM prompts and the dataset generator)."""
    return [label for label, _ in CONDITIONS[condition]["states"]]


def detect_condition(text: str):
    """Return the first condition whose query-name regex appears in the text, else None."""
    low = text.lower()
    for key, spec in CONDITIONS.items():
        if any(re.search(p, low) for p in spec["query_names"]):
            return key
    return None


def mentions_condition(sentence_or_text: str, condition: str) -> bool:
    low = sentence_or_text.lower()
    return any(re.search(p, low) for p in CONDITIONS[condition]["keywords"])


def classify_sentence(sentence: str, condition: str):
    """Return the state label for one sentence, or None if the lexicon does not recognise it."""
    low = sentence.lower()
    for label, patterns in CONDITIONS[condition]["states"]:
        if any(re.search(p, low) for p in patterns):
            return label
    return None


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
