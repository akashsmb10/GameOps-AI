# Architecture

## Layers and responsibilities

| Layer | Code | Responsibility |
|---|---|---|
| Ingestion | `src/ingestion/` | `demo_generator.py` (synthetic), `steam.py` (real connector), `loader.py` (schema check) |
| Preprocessing | `src/preprocessing/clean.py` | dedup, timestamp parsing, language check, spam, invalid values; flags not drops; DQ report |
| NLP | `src/nlp/` | `rules.py` (weak labeler), `classifiers.py` (4 complaint models + sentiment, selection, evaluation), `embeddings.py` (ST / spaCy / TF-IDF-SVD) |
| Analytics | `src/analytics/store.py` | read-only DuckDB; parameterised SQL; Wilson CIs; two-proportion tests; emerging issues |
| RAG | `src/rag/` | `index.py` (chunking, FAISS + TF-IDF, fusion, coverage), `retriever.py` (filters, gate), `answer.py` (citations), `calibrate.py`, `select_mode.py` |
| Agent | `src/agents/` | `tools.py` (5 tools), `guard.py`, `planner.py`, `graph.py` (LangGraph), `llm.py` (Ollama) |
| Evaluation | `src/evaluation/` | `rag_eval.py`, `agent_eval.py`, `agent_heldout.py`, `fake_llm.py` |
| Monitoring | `src/monitoring/telemetry.py` | SQLite telemetry: requests, retrievals, agent runs |
| Serving | `app/` | FastAPI app, schemas, startup loading, middleware |
| UI | `dashboard/` | Streamlit multipage app over the same services |
| Fine-tuning | `src/finetuning/lora_demo.py` | optional LoRA demo, independent of the platform |

## Build flow (`scripts/build_pipeline.py`)
1. Load raw reviews (demo CSV or Steam CSVs) and update notes.
2. Clean → `data/processed/reviews_clean.parquet` + `data_quality_report.json`.
3. Train/select/evaluate complaint and sentiment models → `artifacts/nlp/`.
4. Predict categories and model sentiment for every included review.
5. Write `artifacts/gameops.duckdb` (tables `reviews`, `update_notes`), atomically (temp file + rename).
6. Chunk + embed reviews and docs, build FAISS and lexical indexes → `artifacts/index/`.
7. Select the fusion mode on the dev question set; record the no-answer gate.
8. Write `artifacts/pipeline_manifest.json` (dataset, embedder, model, mode, timestamp).

## Request flow (agent)
`POST /api/v1/agent/query` → `ask()` → LangGraph:
`guard` → (`llm_step` ⇄ `llm_tools`) or (`rule_plan` → `run_tools`) → `synthesize` → `finalize`.
Each tool call: Pydantic validation → thread-pool execution with timeout → structured result or structured error.
`synthesize` keeps only citations that refer to evidence returned by tools; everything is appended to the audit log
and the telemetry DB.

## Design decisions
* **Read-only by construction**: DuckDB opened with `read_only=True`; no write endpoint exists; tools cannot run SQL.
* **One code path for API, dashboard and agent** (`src/agents/tools.py` caches the store and retriever), so numbers agree.
* **Flags instead of deletes** in cleaning; exclusions are explainable.
* **Embedder recorded with the index**; loading an index with a different embedder raises an error.
* **No microservices**: one process for the API, one for the dashboard; Ollama is optional.
