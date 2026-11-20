# EA EADP Data Science Intern - requirement alignment

Status uses evidence in this repository only. "Partial" means implemented but not fully verified or not at the scale
the requirement implies.

| # | Requirement | Implemented feature | Files | Evidence | Status | Limitations |
|---|---|---|---|---|---|---|
| 1 | Python programming | entire platform | `src/`, `app/`, `dashboard/` | 66 tests, ruff clean | Implemented | - |
| 2 | Machine learning & NLP | multi-label complaint classification (4 methods), sentiment model, embeddings | `src/nlp/` | `artifacts/nlp/nlp_metrics.json`, `test_nlp_rag.py` | Implemented | trained/evaluated on synthetic labels |
| 3 | Scikit-learn / PyTorch | sklearn models; PyTorch + PEFT LoRA demo | `src/nlp/classifiers.py`, `src/finetuning/` | `test_finetuning.py`, `finetune_results.json` | Implemented | PyTorch only in the optional module |
| 4 | Cleaning & processing large datasets | flag-based cleaning, DQ report, DuckDB | `src/preprocessing/clean.py`, `src/analytics/store.py` | `data_quality_report.json`, `test_preprocessing.py` | Partial | ~12.5k rows tested; no distributed processing |
| 5 | Developing custom ML applications | API + dashboard + agent around the models | `app/`, `dashboard/`, `src/agents/` | `test_api.py`, `test_dashboard.py` | Implemented | - |
| 6 | Model evaluation | held-out phrasing splits, per-category P/R/F1, confusion matrix, RAG & agent suites, ablation | `src/evaluation/`, `src/nlp/classifiers.py` | `docs/RAG_EVALUATION.md`, `docs/AGENT_EVALUATION.md` | Implemented | synthetic ground truth |
| 7 | Bias identification & mitigation | source-vs-model sentiment comparison by game; self-selection caveats; validation-design correction | `dashboard/views/sentiment.py`, docs | dashboard, walkthrough | Partial | no demographic data (by design); no formal fairness metrics |
| 8 | Deployment & hosting | FastAPI, Dockerfile, compose, AWS guide | `app/`, `Dockerfile`, `docs/DEPLOYMENT.md` | local API runs; `docker compose config` valid | Partial | image build and cloud deploy not executed |
| 9 | Production monitoring | request/latency/error, retrieval, tool success, citation validity, freshness | `src/monitoring/telemetry.py` | `/api/v1/monitoring`, `test_monitoring.py` | Partial | local demo telemetry, not production traffic |
| 10 | Cloud familiarity | App Runner / ECS Fargate guide | `docs/DEPLOYMENT.md` | document | Partial | no hands-on deployment |
| 11 | Data structures & algorithms | FAISS exact search, reciprocal-rank fusion, sparse lexical scoring, sliding-window chunking, Wilson/z-tests | `src/rag/index.py`, `src/analytics/store.py` | tests | Implemented | - |
| 12 | Design patterns & CI/CD | service layer, Pydantic schemas, tool registry, state graph; GitHub Actions (lint, security, tests, docker) | `.github/workflows/ci.yml` | workflow file; same commands pass locally | Partial | workflow not yet run on GitHub |
| 13 | LLM architecture | transformer/attention/LoRA explainer; Ollama integration | `docs/FINETUNING_GUIDE.md`, `src/agents/llm.py` | docs; LLM path tests (scripted) | Partial | no real LLM run here |
| 14 | LLM training & fine-tuning | LoRA pipeline with masking, before/after eval, adapter saving | `src/finetuning/lora_demo.py` | `finetune_results.json` (tiny model) | Partial | pretrained-model run unverified |
| 15 | AI agents | LangGraph agent, 5 validated tools, guard, limits, audit log, 43+10 test cases | `src/agents/` | `agent_eval.json` (43/43), held-out 8/10 | Implemented | real-LLM tool selection not measured |
| 16 | Scalable & reliable systems | read-only stateless API, timeouts, structured errors, health/readiness, load test | `app/main.py`, `deployment/locustfile.py` | `artifacts/eval/load_test.json` | Partial | single-machine benchmark (114 req/s on 2 vCPU) |
| 17 | Communicating business insights | executive overview insights, update-impact page, causal caveats, evidence citations | `dashboard/views/` | screenshots in `docs/screenshots/` | Implemented | synthetic insights |
