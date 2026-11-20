# Project walkthrough

## 1. The question the project answers
"Why are players unhappy, what is getting worse, and what should we investigate first?" - answered with counts,
confidence intervals, significance tests and quoted reviews, through a dashboard, an API and an agent.

## 2. Data
No external source was reachable from the build environment, so the default dataset is a **synthetic** generator
(5 fictional games, 6 months). It was designed to be honest about itself: labels are kept separately, defects are
injected for the cleaning step, and update effects are built in so analysis has something real to find. A Steam
connector (official public endpoints) is included for local real-data runs; review text is not redistributed.

## 3. Cleaning
Flags rather than deletes. 12,528 raw rows → 11,971 included: 120 duplicate ids, 48 spam, 360 non-English,
17 bad timestamps, 15 blank. A bug found here: English sentences containing "no" were detected as Spanish, which was
fixed by requiring zero English stop-words before overriding the source language.

## 4. Complaint models - the most important lesson
First design: train on one set of phrasings, validate on a random sample of the same phrasings, test on new phrasings.
Result: TF-IDF won validation with micro-F1 **0.9997** and scored **0.29** on test - it had memorised templates.
Fix: validation uses its own unseen phrasings. Then validation picked TF-IDF + rules (0.943), which tested at 0.788,
tied with plain keyword rules (0.795). Lesson: **the validation set must resemble deployment, not training.**

## 5. Analytics
DuckDB, read-only, parameterised SQL. Shares come with Wilson 95% intervals; period comparisons with two-proportion
z-tests; windows under 30 reviews are flagged. Example finding: Starfall Arena update 2.1 coincides with server
complaints rising 10.7% → 32.6% of reviews (p < 0.0001) - which exists because the generator put it there.

## 6. Retrieval
FAISS + TF-IDF hybrid with metadata filters. Two evidence-driven decisions:
* fusion mode chosen on a dev set (sum of scores, 0.74 vs dense 0.46);
* the no-answer gate changed from a cosine threshold (impossible with averaged vectors) to vocabulary coverage, which
  refuses all off-topic evaluation questions.

## 7. Agent
LangGraph state graph: guard → plan (LLM tool calling or deterministic planner) → validated, time-limited tool
execution → synthesis with citation checking → audit. The deterministic path is labelled so nobody mistakes
templates for an LLM. The evaluation suite found four real bugs before reaching 43/43; ten held-out questions scored
8/10 and their failures are documented.

## 8. Serving, monitoring, quality
FastAPI (10 endpoints, JSON logs, 422/503 handling), telemetry in SQLite, Streamlit dashboard (7 pages) on the same
services, 66 tests, ruff/bandit/pip-audit (dependency upgrades removed 85 known vulnerabilities), Locust benchmark.

## 9. What is not proven
Real-world accuracy (no real labelled data), real LLM behaviour, Sentence-Transformers quality, pretrained-model
fine-tuning, the Docker image build and any cloud deployment.

## 10. Talking about it in 60 seconds
"GameOps AI analyses game reviews end-to-end. A cleaning pipeline flags defects and reports them. Complaint
categories come from a model selected on unseen wording - I learned the hard way that validating on training-like
data gave 0.9997 and then 0.29 on new text. Analytics run as read-only SQL with confidence intervals. A LangGraph
agent chooses among five validated, read-only tools, cites every claim and refuses to call correlations causes. It is
served by FastAPI and a Streamlit dashboard, with 66 tests and documented evaluations - on synthetic data, which I
state up front."
