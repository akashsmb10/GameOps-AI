"""Run every evaluation suite and print a one-screen summary."""
import json

import _bootstrap  # noqa: F401

from src.evaluation import agent_eval, rag_eval

if __name__ == "__main__":
    rag = rag_eval.evaluate()
    agent = agent_eval.evaluate()
    print(json.dumps({"rag": {k: rag[k] for k in ("questions", "review_precision_at_k", "doc_hit_at_3", "citation_correctness",
                                                  "groundedness", "unanswerable_refused", "answerable_answered")},
                      "agent": {k: agent[k] for k in ("cases", "passed", "pass_rate")}}, indent=2))
