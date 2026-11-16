# Time-Aware Agentic RAG for Resolving Contradictory Longitudinal Patient Records

B.Tech Project-I, Delhi Technological University (DTU/CSE/BTP/GP043). Code that implements the plan in the
Detailed Research Plan: a RAG pipeline that treats time as **validity**, not just a ranking score, and an
arbitration agent that suppresses superseded diagnoses **before** answer generation.

> Everything in `data/` is **synthetic**. No real patient data is used or included.

## How the code maps to the report

| Report item | Where it lives |
|---|---|
| **G1 / RQ1** NER timestamps as hard metadata + query-aware filter | `src/ner_ingestion.py`, `src/vector_store.py`, `src/temporal_filter.py` |
| **G2 / RQ2** arbitration agent suppressing superseded diagnoses (auditable) | `src/arbitrator.py`, `src/arbitrator_graph.py` (LangGraph) |
| **G3 / RQ3** current-state accuracy *with* latency + token overhead | `evaluate.py`, `src/metrics.py` |
| Baseline 1: naive RAG; Baseline 2: soft time-decay re-ranking | `src/baselines.py` |
| Phase 2 dataset | `scripts/generate_dataset.py` (synthetic, labelled) |

```
notes (free text) -> spaCy+dateparser -> doc_date metadata -> embeddings + metadata in ChromaDB
query -> parse (condition, as-of date) -> retrieve with HARD date bound -> chronological sort
      -> [router: >=2 notes mention the condition?] -> arbitrate (extract states -> suppress superseded + audit log)
      -> generate answer
```

## Setup (Python 3.12+)

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env                                        # then edit
pytest -q                                                   # 25 tests, runs offline
```
`requirements.txt` includes `sentence-transformers` (PyTorch, large). For a quick try set `EMBEDDER=tfidf`.

## Run

```bash
python scripts/generate_dataset.py                          # (re)build data/ ; seed 42, 40 patients
python eval_extraction.py                                   # date-extraction accuracy (RQ1 support)
python evaluate.py --embedder tfidf                         # all 5 pipelines on the TEST split, offline
python scripts/demo.py --query-id Q0003 --embedder tfidf    # one query through every pipeline + audit log
```

Pipelines compared (`--pipelines naive,decay,filter,arbitrator,full`):

| name | what it is |
|---|---|
| `naive` | top-K cosine similarity (time-blind) |
| `decay` | similarity blended with exponential half-life decay (time as a score) |
| `filter` | ablation: hard date bound + chronological order, no arbitrator (G1 only) |
| `arbitrator` | ablation: arbitrator without the date bound (G2 only) |
| `full` | proposed system: filter + arbitrator |

Outputs: a table in the terminal and `results/summary.csv`, `results/per_query.csv`, `results/run_config.json`.

### Using a real LLM
Edit `.env` (`LLM_PROVIDER=groq|gemini|ollama`, key/model), install the matching package from `requirements.txt`, then e.g.
```bash
python evaluate.py --llm groq --arbitrator-mode hybrid --limit 40      # start small: free tiers are rate-limited
```
* **Reader**: with an LLM the answer is produced by the model (`src/generator.py`). Without one, an offline *rule reader*
  is used. It is only a proxy that understands the lexicon's wording; **report LLM-reader numbers**.
* **Arbitrator modes**: `rules` (no LLM calls), `hybrid` (LLM only for notes the rules cannot classify), `llm`
  (LLM extracts the state from every retrieved note). Token/latency overhead (RQ3) is only meaningful in `hybrid`/`llm`.
* The LLM code paths are unit-tested with a fake model, but were **not run against a real provider** when this
  repo was generated. Test them yourselves and fix model names if your provider has changed them.

## Metrics (see `src/metrics.py`)
`clean_context` = the note stating the correct state is in the final context **and** no wrong-time note is
(wrong-time = superseded or dated after the query's as-of date). `exposure` = both correct and wrong-time notes reach
the generator (the "2020 vs 2024" scenario). `CRR` = of the queries whose retrieved set contained such a conflict, the
share that were clean after the pipeline ran (RQ2). `accuracy` = answer state equals gold state. Cost: latency,
tokens sent to the generator, tokens spent by LLM arbitration, LLM calls (RQ3). `evaluate.py` also prints paired
bootstrap 95% confidence intervals of `full - baseline`.

## Read this before putting numbers in the report
1. **Synthetic data is easy and partly circular.** The rule arbitrator uses the same condition vocabulary the
   generator draws from (70% of mentions use that wording, 30% are paraphrases it cannot read). High scores on this
   data do not show clinical validity. Use it to debug and for relative comparisons, then evaluate on MIMIC-IV
   notes (PhysioNet credentialed access) or another labelled corpus, and say so in the report.
2. **The date filter alone cannot detect supersession.** It removes *future* evidence. Superseded-but-past notes need
   the arbitrator. This is why the `filter` ablation helps as-of queries but not "current" queries.
3. **Retrieval recall bounds everything.** If the correct note is not retrieved, no arbitrator can recover it.
4. **Tune on `--split dev`, report on `--split test`.** Baseline hyper-parameters (`DECAY_HALF_LIFE_DAYS`, `DECAY_ALPHA`)
   should be chosen on dev, not on the queries you report.
5. **Patient-scoped search**: every pipeline searches within one patient's record. Fair across pipelines, but real
   systems must first identify the patient.
6. **spaCy-sm limitations**: the small model mislabels some full dates (ISO as CARDINAL, "9 February 2026" as EVENT),
   so full-date formats are matched by regex first. This was found on our own small hand-labelled sample, so that
   sample is no longer an unbiased test: add new hand-labelled notes to `data/extraction_gold.json`.
7. Token counts are estimated (~4 characters/token) unless the provider returns usage data.
8. This is research code, not a clinical tool. Do not use it for patient care.

## Credits and licence
See `THIRD_PARTY_NOTICES.md`. Code in this repository: MIT (`LICENSE`).
