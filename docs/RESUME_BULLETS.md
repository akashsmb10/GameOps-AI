# Resume bullets (only verified work; demo data is synthetic)

* Built **GameOps AI**, an agentic player-feedback platform in Python/SQL: a **LangGraph** agent selects among 5 Pydantic-validated, read-only tools (DuckDB analytics, FAISS hybrid retrieval, update notes), verifies every citation, and passed **43/43** deterministic evaluation cases and **8/10** held-out cases.
* Implemented **hybrid RAG** (FAISS dense + TF-IDF lexical fusion, metadata filters, vocabulary-coverage no-answer gate) reaching **0.956 precision@5**, **1.0 citation correctness** and **6/6 off-topic refusals** on a 30-question evaluation; designed leakage-free model selection with disjoint validation phrasing.
* Shipped a **FastAPI** service (10 endpoints, structured logging, SQLite telemetry) and a 7-page **Streamlit/Plotly** dashboard, with **66 pytest tests**, GitHub Actions CI, a clean pip-audit, and a Locust benchmark (113.7 req/s, 0 failures on 2 vCPU).

Notes for honesty in interviews: the dataset is synthetic (a Steam connector exists but was not run); LLM answers,
Sentence-Transformers and pretrained-model LoRA were implemented but not executed in the build environment.
