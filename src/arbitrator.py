"""Clinical State Arbitrator (addresses G2 / RQ2).

Two clearly separated steps, so every decision is auditable:

  1. STATE EXTRACTION  - for each retrieved note, which state of the target condition does it assert?
        rules : lexicon in medical_lexicon.py (no LLM calls)
        llm   : one batched LLM call returns a state label per note
        hybrid: rules first; the LLM is asked only about notes that mention the condition but the
                rules could not classify
  2. SUPERSESSION RESOLUTION (deterministic, no LLM) - the newest classified note defines the current
        state; older notes asserting a different state are suppressed *before generation*, and every
        suppression is written to an audit log (note id, dates, old state -> new state, reason).

Safety rules (abstain rather than risk dropping the true current note):
  * If a note that mentions the condition but could not be classified is NEWER than the newest
    classified note, the arbitrator abstains from suppressing anything and flags it.
  * Notes older than the current note whose state could not be classified are dropped only if
    `drop_older_unclassified=True` (default), with reason "older_unclassified".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm import TokenMeter, call_llm, extract_json
from .medical_lexicon import (CONDITIONS, classify_sentence, condition_states, mentions_condition,
                              split_sentences)
from .schema import Chunk


@dataclass
class Observation:
    note_id: str
    date: object
    mentions: bool                 # does the note talk about the target condition at all?
    state: Optional[str]           # asserted state, or None if unknown
    method: str                    # "rules" | "llm"


@dataclass
class ArbitrationDecision:
    kept: list                                   # chunks to pass on to the generator
    suppressed: list = field(default_factory=list)   # audit log (list of dicts)
    current_state: Optional[str] = None
    conflict_detected: bool = False
    abstained: bool = False
    notes: list = field(default_factory=list)    # human-readable remarks (unclassified notes, abstentions)


# ---------------------------------------------------------------- step 1: state extraction
def rule_observation(chunk: Chunk, condition: str) -> Observation:
    """Classify a note with the lexicon: the LAST classified sentence about the condition wins."""
    mentions, state = False, None
    for sent in split_sentences(chunk.text):
        if mentions_condition(sent, condition):
            mentions = True
            label = classify_sentence(sent, condition)
            if label is not None:
                state = label
    return Observation(chunk.note_id, chunk.doc_date, mentions, state, "rules")


def llm_states(chunks: list, condition: str, llm, meter: Optional[TokenMeter]) -> dict:
    """One batched call: {note_id: state label | None}. Unknown labels are mapped to None."""
    allowed = condition_states(condition)
    display = CONDITIONS[condition]["display"]
    notes_txt = "\n".join(f"[{c.note_id}] {c.text}" for c in chunks)
    prompt = (
        f"You read clinical notes. For each note, decide which state of the patient's {display} the note "
        f"asserts at the time it was written.\n"
        f"Allowed states: {allowed}. Use \"unknown\" if the note does not state the {display} status, "
        f"and \"not_mentioned\" if the note is not about {display} at all.\n"
        f"Return ONLY JSON of the form {{\"<note_id>\": \"<state>\", ...}}.\n\nNotes:\n{notes_txt}\n"
    )
    reply = call_llm(llm, prompt, meter, role="arbitrator")
    data = extract_json(reply) or {}
    out = {}
    for c in chunks:
        label = data.get(c.note_id)
        out[c.note_id] = label if label in allowed else None
    return out


# ---------------------------------------------------------------- step 2: deterministic resolution
def resolve_supersession(chunks: list, observations: list, drop_older_unclassified: bool = True) -> ArbitrationDecision:
    obs = {o.note_id: o for o in observations}
    classified = sorted([o for o in observations if o.state], key=lambda o: (o.date, o.note_id))
    if not classified:
        return ArbitrationDecision(kept=list(chunks), notes=["no note could be classified; nothing suppressed"])

    newest = classified[-1]
    same_day = [o for o in classified if o.date == newest.date and o.state != newest.state]
    unclassified = [o for o in observations if o.mentions and not o.state]
    newer_unclassified = [o for o in unclassified if o.date > newest.date]
    distinct_states = {o.state for o in classified}
    decision = ArbitrationDecision(kept=[], current_state=newest.state,
                                   conflict_detected=len(distinct_states) > 1)

    if same_day:
        decision.abstained = True
        decision.notes.append(f"same-day conflict on {newest.date.date()}: cannot order notes; nothing suppressed")
        decision.kept = list(chunks)
        return decision
    if newer_unclassified:
        decision.abstained = True
        ids = ", ".join(o.note_id for o in newer_unclassified)
        decision.notes.append(f"newer note(s) [{ids}] mention the condition but could not be classified; "
                              f"abstaining to avoid trusting an outdated state")
        decision.kept = list(chunks)
        return decision

    for c in chunks:
        o = obs[c.note_id]
        if not o.mentions or o.note_id == newest.note_id or o.date > newest.date:
            decision.kept.append(c)
            continue
        if o.state and o.state != newest.state:
            reason = "superseded"
        elif not o.state and drop_older_unclassified:
            reason = "older_unclassified"
        else:
            decision.kept.append(c)                      # same state (corroborating) or unclassified-kept
            continue
        decision.suppressed.append({
            "note_id": o.note_id, "date": o.date.date().isoformat(),
            "old_state": o.state, "superseded_by": newest.note_id,
            "new_state": newest.state, "new_date": newest.date.date().isoformat(),
            "reason": reason, "method": o.method,
        })
    for o in unclassified:
        if o.date < newest.date:
            decision.notes.append(f"{o.note_id}: mentions the condition, state not recognised")
    return decision


class Arbitrator:
    def __init__(self, mode: str = "rules", llm=None, meter: Optional[TokenMeter] = None,
                 drop_older_unclassified: bool = True):
        if mode not in ("rules", "hybrid", "llm"):
            raise ValueError("mode must be rules | hybrid | llm")
        if mode in ("hybrid", "llm") and llm is None:
            mode = "rules"            # no LLM available: degrade gracefully (recorded by caller via method field)
        self.mode, self.llm, self.meter = mode, llm, meter
        self.drop_older_unclassified = drop_older_unclassified

    def has_candidates(self, chunks: list, condition: Optional[str]) -> bool:
        """Cheap router test: can there be a conflict? (>= 2 notes that mention the condition)."""
        if condition is None:
            return False
        if self.mode == "llm":
            return len(chunks) >= 2
        return sum(1 for c in chunks if mentions_condition(c.text, condition)) >= 2

    def arbitrate(self, chunks: list, condition: Optional[str]) -> ArbitrationDecision:
        if condition is None or not chunks:
            return ArbitrationDecision(kept=list(chunks), notes=["no recognised condition in query"])

        if self.mode == "llm":
            states = llm_states(chunks, condition, self.llm, self.meter)
            observations = [Observation(c.note_id, c.doc_date, states[c.note_id] is not None,
                                        states[c.note_id], "llm") for c in chunks]
        else:
            observations = [rule_observation(c, condition) for c in chunks]
            if self.mode == "hybrid":
                todo = [c for c, o in zip(chunks, observations) if o.mentions and o.state is None]
                if todo:
                    states = llm_states(todo, condition, self.llm, self.meter)
                    for o in observations:
                        if o.note_id in states and states[o.note_id] is not None:
                            o.state, o.method = states[o.note_id], "llm"
        return resolve_supersession(chunks, observations, self.drop_older_unclassified)
