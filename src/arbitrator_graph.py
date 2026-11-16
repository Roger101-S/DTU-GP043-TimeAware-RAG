"""LangGraph state machine for the proposed pipeline (Phase 3 + 4 of the report).

    parse_query -> retrieve (+hard temporal filter) -> chronological_sort -> [conflict router]
                                                                   |-- candidates exist --> arbitrate --\
                                                                   '-- otherwise --------------------> generate -> END

Flags make the ablations in the evaluation possible with one implementation:
    use_filter=True,  use_arbitrator=True   -> "full"        (proposed system)
    use_filter=True,  use_arbitrator=False  -> "filter"      (G1 only)
    use_filter=False, use_arbitrator=True   -> "arbitrator"  (G2 only; ignores the query's as-of date)

The conditional edge is the 'agentic' part: the (possibly LLM-backed) arbitrator is only invoked when two or
more retrieved notes mention the queried condition, i.e. when a contradiction is actually possible.
"""
from __future__ import annotations

import time
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .arbitrator import ArbitrationDecision, Arbitrator
from .generator import Answerer, context_token_estimate
from .schema import ParsedQuery, PipelineResult, Query
from .temporal_filter import chronological_sort, parse_query, temporal_bounds


class GraphState(TypedDict, total=False):
    query: Query
    parsed: ParsedQuery
    retrieved: list
    context: list
    decision: Optional[ArbitrationDecision]
    answer: str
    answer_state: Optional[str]
    trace: list


class TimeAwareAgenticRAG:
    def __init__(self, store, embedder, extractor, answerer: Answerer, arbitrator: Arbitrator,
                 top_k: int, reference_date, use_filter: bool = True, use_arbitrator: bool = True,
                 name: Optional[str] = None):
        self.store, self.embedder, self.extractor = store, embedder, extractor
        self.answerer, self.arbitrator = answerer, arbitrator
        self.top_k, self.reference_date = top_k, reference_date
        self.use_filter, self.use_arbitrator = use_filter, use_arbitrator
        self.name = name or ("full" if use_filter and use_arbitrator else
                             "filter" if use_filter else "arbitrator")
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _parse(self, s: GraphState) -> GraphState:
        parsed = parse_query(s["query"].text, self.extractor, self.reference_date)
        return {"parsed": parsed, "trace": ["parse_query"]}

    def _retrieve(self, s: GraphState) -> GraphState:
        q, parsed = s["query"], s["parsed"]
        emb = self.embedder.encode([q.text])[0]
        max_ts = temporal_bounds(parsed)[1] if self.use_filter else None   # HARD bound inside the DB query
        hits = self.store.query(emb, q.patient_id, self.top_k, max_ts=max_ts)
        chunks = []
        for c, sim in hits:
            c.score = sim
            chunks.append(c)
        return {"retrieved": chunks, "context": list(chunks), "trace": s["trace"] + ["retrieve"]}

    def _sort(self, s: GraphState) -> GraphState:
        if self.use_filter:
            ordered = chronological_sort(s["retrieved"])
            return {"retrieved": ordered, "context": list(ordered), "trace": s["trace"] + ["chronological_sort"]}
        return {"trace": s["trace"] + ["chronological_sort(skipped)"]}

    def _route(self, s: GraphState) -> str:
        if self.use_arbitrator and self.arbitrator.has_candidates(s["retrieved"], s["parsed"].condition):
            return "arbitrate"
        return "generate"

    def _arbitrate(self, s: GraphState) -> GraphState:
        decision = self.arbitrator.arbitrate(s["retrieved"], s["parsed"].condition)
        return {"decision": decision, "context": decision.kept, "trace": s["trace"] + ["arbitrate"]}

    def _generate(self, s: GraphState) -> GraphState:
        parsed = s["parsed"]
        ans, state = self.answerer.answer(s["query"].text, parsed.condition,
                                          parsed.as_of.date().isoformat(), s["context"])
        return {"answer": ans, "answer_state": state, "trace": s["trace"] + ["generate"]}

    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("parse_query", self._parse)
        g.add_node("retrieve", self._retrieve)
        g.add_node("chronological_sort", self._sort)
        g.add_node("arbitrate", self._arbitrate)
        g.add_node("generate", self._generate)
        g.add_edge(START, "parse_query")
        g.add_edge("parse_query", "retrieve")
        g.add_edge("retrieve", "chronological_sort")
        g.add_conditional_edges("chronological_sort", self._route, {"arbitrate": "arbitrate", "generate": "generate"})
        g.add_edge("arbitrate", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # ------------------------------------------------------------------ public API
    def run(self, q: Query) -> PipelineResult:
        t0 = time.perf_counter()
        out = self.graph.invoke({"query": q})
        decision: Optional[ArbitrationDecision] = out.get("decision")
        return PipelineResult(
            query_id=q.query_id, pipeline=self.name, retrieved=out["retrieved"], context=out["context"],
            answer=out["answer"], answer_state=out["answer_state"],
            suppressed=decision.suppressed if decision else [],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            context_tokens=context_token_estimate(out["context"]), arbitration_tokens=0, llm_calls=0,
            arbitrated=decision is not None, current_state=decision.current_state if decision else None,
        )
