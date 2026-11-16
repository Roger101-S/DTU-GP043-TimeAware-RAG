# File-by-file explainer (for the viva)

Read each file once and run it yourself. Be ready to say *what it does*, *why this design*, and *what it cannot do*.

## src/

**`schema.py`** - Data classes only (`Note`, `Chunk`, `Query`, `ParsedQuery`, `PipelineResult`). A `Note` is raw text plus a
system filing date. A `Chunk` adds the extracted `doc_date`. One note = one chunk (notes are short).

**`ner_ingestion.py`** (G1 / RQ1) - Turns free text into dates. Full dates (ISO, "3rd March 2021", "Mar 3, 2021",
12/05/2023) are matched by regex first; spaCy `DATE` entities cover the rest; `dateparser` converts every span to a
datetime (`DATE_ORDER=DMY` fixes 03-04 ambiguity). The *document date* is the first absolute date near the top of the note.
Relative phrases ("3 months ago") are resolved against that date, not today. If no date is found, the system filing date is
used (`date_source="fallback"`); with neither, the note is excluded and counted.
*Why regex first?* Measured: spaCy-sm labelled ISO dates CARDINAL and split "3rd March 2021". *Limitation:* year-only
mentions become 1 Jan; only the visit date is used for filtering.

**`vector_store.py`** - ChromaDB (as in the report) with `patient_id` and `date_ts` stored as metadata. The date filter is a
`where` clause inside the database query, i.e. a **hard constraint**, not a score. A NumPy store with the same interface is
the fallback.

**`embedder.py`** - `minilm` (sentence-transformers all-MiniLM-L6-v2) or `tfidf` (offline, deterministic). The TF-IDF
version drops filler words because without that, short routine notes outranked condition notes.

**`temporal_filter.py`** - `parse_query` finds the condition and the as-of date (explicit "as of June 2021" -> 30 June 2021,
else the fixed reference date 2026-01-01). `chronological_sort` orders evidence oldest to newest.
*Limitation:* it removes future notes only; it cannot know a past note was superseded.

**`medical_lexicon.py`** - Five conditions with ordered regex patterns per state ("no history of..." is checked before
"stage 2..."). This is the transparent rule baseline; wording outside it is invisible to the rules.

**`arbitrator.py`** (G2 / RQ2) - Step 1 extracts each note's asserted state (rules / LLM / hybrid). Step 2 is deterministic:
the newest classified note defines the current state; older notes with a *different* state are suppressed and logged
(note id, dates, old -> new state, reason, method). Same-state notes are kept as corroboration. **Safety:** it abstains if a
newer note mentions the condition but could not be classified, or if two notes on the same day disagree. *Why split
extraction from resolution?* The LLM only reads; the decision is plain code, so it is auditable and testable.

**`arbitrator_graph.py`** - The LangGraph state machine: parse -> retrieve -> sort -> router -> (arbitrate) -> generate.
The router calls the arbitrator only when >=2 retrieved notes mention the condition (saves LLM calls: RQ3).
`use_filter` / `use_arbitrator` flags give the `filter`, `arbitrator` and `full` variants from one implementation.

**`baselines.py`** - `NaiveRAG` (top-K cosine) and `SoftTimeDecayRAG`
(`alpha * sim_norm + (1 - alpha) * 0.5 ** (age_days / half_life)`). Neither understands the query's date.

**`generator.py`** - LLM reader (prompt asks for one state label as JSON) and an offline rule reader (proxy: returns the single
state in the context, `CONFLICT` if several, `unknown` if none).

**`llm.py`** - LangChain chat-model factory (Groq / Gemini / Ollama), retry with back-off, token meter, tolerant JSON parser.

**`metrics.py`** - Per-query scoring against gold labels, aggregation, paired bootstrap CIs. Definitions in the README.

**`runner.py`** - Ingests notes, builds every pipeline once, measures LLM calls/tokens per query.

**`config.py`** - Reads `.env`; fixes `REFERENCE_DATE = 2026-01-01` so "current" means the same on every run.

## top level
**`evaluate.py`** runs all pipelines on the same queries and writes the tables. **`eval_extraction.py`** measures date-extraction
accuracy on a hand-written sample and on the synthetic notes. **`scripts/generate_dataset.py`** builds notes, queries and gold
labels (seeded). **`scripts/demo.py`** shows one query through every pipeline with the audit log. **`tests/`** covers date
parsing, hard filtering, decay math, supersession rules, abstention, the LLM paths (fake model) and the full graph.

## Likely examiner questions
* **How do you know a note is superseded?** The newest classified note defines the state; older notes asserting a different
  state are dropped and logged. Abstains when unsure.
* **Why not just rank by recency?** A score cannot say "this stopped being true"; an old note can still be retrieved (our
  `decay` baseline shows it).
* **What if NER extracts the wrong date?** Accuracy is measured (`eval_extraction.py`); fallback to filing date; a hand-labelled
  sample you extend.
* **Is your dataset realistic?** No: it is synthetic and partly matches the rule vocabulary; say so, and evaluate on MIMIC-style notes.
* **What does the arbitrator cost?** `rules`: no LLM calls. `hybrid`/`llm`: extra calls, reported as `llm_calls` and `arb_tok`.
* **Difference from temporal-rag / EHR-RAG / RASTeR?** Timestamps come from NER over free text; supersession is an explicit,
  logged step on clinical diagnoses; accuracy and overhead are reported together.
