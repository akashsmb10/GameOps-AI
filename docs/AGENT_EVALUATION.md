# Agent evaluation

Run: `python -m src.evaluation.agent_eval` (43 cases) and `python -m src.evaluation.agent_heldout` (10 cases).

## Checks (deterministic)
* **tools** - expected tools called, forbidden tools not called, minimum call counts
* **args** - key arguments (game id, period boundaries, sentiment)
* **refusal** - refused with the expected category and no tool calls
* **stats** - every percentage in the answer equals a value returned by a tool (±0.05)
* **citations** - no invalid citations; every cited review exists; quoted text matches the cited review
* **content** - required phrases (premise correction, "no reviews", causal caution)
* special cases call `execute_tool` directly, monkey-patch tools to fail or time out, or replace the LLM with a
  scripted tool-calling model (`src/evaluation/fake_llm.py`)

## Results
| Group | Cases | Passed |
|---|---|---|
| tool_selection | 10 | 10 |
| multi_tool | 7 | 7 |
| conflicting_evidence | 3 | 3 |
| missing_data | 4 | 4 |
| prompt_injection | 5 | 5 |
| invalid_input | 4 | 4 |
| tool_argument_validation | 4 | 4 |
| tool_failure_recovery | 2 | 2 |
| llm_path (scripted model) | 6 | 6 |
| **total** | **45** | **45** |
| **held-out** (written after development, run once) | 10 | **8** |

Held-out failures (reported, not fixed for the report):
* H03 "Did matchmaking complaints … go up in September compared with August?" - the planner's comparison intent
  does not include "go up", so it analysed one period instead of comparing two.
* H08 "Is the review data real or synthetic?" - not routed to `get_data_quality_report`.
Both are wording-coverage limits of the rule-based planner; an LLM planner is the intended fix.

## Development history (disclosed)
The first run passed 36/43. Fixes were made to the code, not the tests:
* empty date windows crashed (`NaN` → int) → COALESCE in SQL;
* repeated calls to the same tool overwrote each other's results → results stored per call;
* planner precedence (data-quality and "what changed" questions) → explicit precedence rules;
* evaluation harness bug: a new scripted model per graph step restarted the script → single instance.

## What the LLM-path cases prove - and do not
They prove the graph lets a model choose tools, executes them with validation and limits, rejects non-approved tools,
strips fabricated citations and falls back to a deterministic summary. They do **not** measure a real LLM's tool
selection or answer quality; that requires running with `OLLAMA_URL` set and repeating this suite.

## First real-LLM run (llama3.2:3b via Ollama, laptop CPU) - manual observation
Question: "What are players complaining about most?"
* The model selected `analyze_complaints` (a sensible tool) but **added `game_id=G001` and a date range that the
  question never mentioned**, narrowing an all-games question to one game. Schema validation could not catch this,
  because G001 is a valid id.
* Its final text contained **no valid citations**, so the citation check replaced it with the deterministic summary
  (the guardrail worked as designed). Latency: 94 s on CPU.
* Fix: **argument grounding** (`ground_args` in `src/agents/graph.py`). Before an LLM-proposed call runs, the game and
  time period are re-extracted from the question; a game_id or date filter with no support in the question is removed
  and reported as a warning. The proposed arguments are kept as `llm_args` for provenance. Two cases (L05, L06)
  reproduce the observed behaviour and the allowed case. The system prompt now also states when to pass filters and
  shows the citation format.
* This is one manual observation, not an evaluation. Running `python -m src.evaluation.agent_eval` with `OLLAMA_URL`
  set measures the real model on all cases.

Note: in one run T05 failed its content check and passed on two immediate reruns; the cause was not reproduced.
