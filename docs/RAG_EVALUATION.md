# RAG evaluation

Run: `python -m src.evaluation.rag_eval` → `artifacts/eval/rag_eval.json` and `rag_eval_cases.csv`.
No LLM judge is used: every metric is a deterministic check.

## Question sets
| Set | Size | Purpose | Relevance |
|---|---|---|---|
| Evaluation - review questions | 18 | "What are <game> players saying about <topic>?" for 18 game×category pairs | generator labels: review of that game whose synthetic label contains the category |
| Evaluation - documentation | 6 | patch contents, metric definitions, data provenance | one known document id each |
| Evaluation - unanswerable | 6 | off-topic (geography, recipes, finance, health…) | must be refused |
| Dev (selection only) | 10 | different pairs and wording | used to pick the fusion mode, never for reported scores |
| Calibration probes | 10 + 10 | in-domain / off-domain phrases | used to check the no-answer gate |

## Results (spaCy `en_core_web_md` vectors, mode = sum)
| Metric | Value |
|---|---|
| Review precision@5 | **0.956** |
| Review hit@5 | 1.000 |
| Documentation hit@3 | 1.000 |
| Citation correctness (cited ids ⊆ retrieved evidence) | **1.000** |
| Groundedness (content sentences with a valid citation) | 0.944 |
| Unsupported-claim rate | 0.056 (one review question was refused, so its answer had no citations) |
| Answer relevance (keyword proxy: asked category named in the answer) | 0.944 |
| Unanswerable refused | **6/6** |
| Answerable answered | 23/24 (0.958) |

### Retrieval ablation (same 18 questions)
| Mode | precision@5 | hit@5 |
|---|---|---|
| dense (spaCy vectors) | 0.444 | 0.833 |
| lexical (TF-IDF) | 0.967 | 1.000 |
| hybrid (reciprocal-rank fusion) | 0.733 | 1.000 |
| **sum of scores (selected on dev)** | **0.956** | 1.000 |

Lexical scores slightly higher than the selected mode on this set, but the mode was chosen on the dev set (sum 0.74 vs
lexical 0.68) to avoid tuning on the evaluation questions. The strong lexical result is partly an artefact of
template-based synthetic text sharing vocabulary with the questions.

## What changed during development (disclosed)
1. **Cosine no-answer threshold → rejected.** With averaged word vectors "what is the capital of france" scored 0.93
   while "crashes" scored 0.47; no threshold separates them. A margin-based threshold also refused real questions on a
   smaller corpus (caught by a unit test).
2. **Vocabulary-coverage gate adopted** (≥ 50% of content words appear in ≥ 2 chunks; 0.5 fixed a priori).
3. **Game names excluded from coverage.** The first coverage version answered only 0.625 of answerable evaluation
   questions because game names never appear in review text; this bug was found through the evaluation set and fixed.
   The reported numbers are therefore after one evaluation-informed fix.

## Limitations
* Relevance labels come from the synthetic generator; real-review relevance needs human judgements.
* LLM-generated answers were not evaluated (no LLM server); answers are extractive.
* Citation correctness of 1.0 is guaranteed by construction for extractive answers; for LLM answers the same checker
  removes invalid citations and reports them.
