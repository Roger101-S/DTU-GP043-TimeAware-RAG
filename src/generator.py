"""Answer generation. Two readers:

  * LLM reader   : prompts the configured chat model with the context and asks for a state label.
  * Rule reader  : offline PROXY. It reads the context with the same lexicon and answers with the single
                   state it finds, "CONFLICT: a / b" if the context asserts several states, or "unknown".
                   It exists so the whole pipeline can be tested without an API key. It only understands
                   the lexicon's canonical wording, so absolute accuracy under it is NOT comparable to an
                   LLM; use it to compare pipelines, and use the LLM reader for the numbers you report.
"""
from __future__ import annotations

from typing import Optional

from .llm import TokenMeter, approx_tokens, call_llm, extract_json
from .medical_lexicon import CONDITIONS, condition_states, mentions_condition, split_sentences, classify_sentence


def context_states(chunks: list, condition: str) -> list:
    """States asserted by the context chunks for one condition (rule-based), in input order."""
    found = []
    for c in chunks:
        state = None
        for sent in split_sentences(c.text):
            if mentions_condition(sent, condition):
                label = classify_sentence(sent, condition)
                if label:
                    state = label
        if state:
            found.append(state)
    return found


class Answerer:
    def __init__(self, llm=None, meter: Optional[TokenMeter] = None):
        self.llm, self.meter = llm, meter

    @property
    def reader(self) -> str:
        return "llm" if self.llm is not None else "rules"

    def answer(self, query_text: str, condition: Optional[str], as_of_iso: str, context: list):
        """Returns (answer_text, state_label_or_None)."""
        if condition is None:
            return "Condition not recognised in the query.", None
        if self.llm is None:
            states = context_states(context, condition)
            distinct = sorted(set(states))
            if not distinct:
                return "unknown", "unknown"
            if len(distinct) > 1:
                return "CONFLICT: " + " / ".join(distinct), "CONFLICT"
            return distinct[0], distinct[0]

        allowed = condition_states(condition)
        display = CONDITIONS[condition]["display"]
        notes_txt = "\n".join(f"Note {i + 1}: {c.text}" for i, c in enumerate(context))
        prompt = (
            f"Use ONLY the clinical notes below to state the patient's {display} status as of {as_of_iso}.\n"
            f"Choose exactly one state from {allowed}. If the notes disagree, choose the state that is "
            f"valid at {as_of_iso}.\nReturn ONLY JSON: {{\"state\": \"<state>\", \"explanation\": \"<one sentence>\"}}\n\n"
            f"{notes_txt}\n\nQuestion: {query_text}\n"
        )
        reply = call_llm(self.llm, prompt, self.meter, role="generator")
        data = extract_json(reply) or {}
        state = data.get("state")
        return reply.strip(), (state if state in allowed else "unparsed")


def context_token_estimate(context: list) -> int:
    return sum(approx_tokens(c.text) for c in context)
