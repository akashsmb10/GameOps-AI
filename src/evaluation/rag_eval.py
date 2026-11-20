"""Deterministic RAG evaluation (no LLM judge).

Metrics
* Precision@5 / Hit@5 / Recall@5(capped): relevant = included reviews of the asked game whose
  SYNTHETIC generator label contains the asked category. Recall@K is capped:
  |relevant in top K| / min(K, |relevant|).
* No-answer accuracy: unanswerable questions must be refused; answerable ones must not.
* Citation correctness: cited ids that exist in the retrieved evidence / all cited ids.
* Groundedness: share of content sentences in the answer carrying at least one valid citation.
  Unsupported-claim frequency = 1 - groundedness. The closing disclaimer is not counted.
* Answer relevance (proxy): the asked category appears among the categories summarised in
  the answer (extractive mode) - a keyword check, not a semantic judgement.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.config import CATEGORY_LABELS, get_settings
from src.rag.answer import answer, validate_citations
from src.rag.retriever import Retriever

from .rag_questions import CATEGORY_PHRASES, DOC_QUESTIONS, REVIEW_PAIRS, UNANSWERABLE

META_PREFIXES = ("These are retrieved examples",)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"\n+", text) if s.strip() and not s.strip().startswith(META_PREFIXES)]


def evaluate(k: int = 5, modes: tuple = ("dense", "lexical", "hybrid", "sum")) -> dict:
    s = get_settings()
    r = Retriever()
    labels = pd.read_parquet(s.raw_dir / "labels_synthetic.parquet").set_index("review_id")
    games = dict(r.reviews.meta.groupby("game_id")["game_name"].first())
    rows, ablation = [], {m: [] for m in modes}
    for gid, cat in REVIEW_PAIRS:
        q = f"What are {games[gid]} players saying about {CATEGORY_PHRASES[cat]}?"
        allowed = r._allowed(gid, None, None, None)
        relevant = set(labels.index[labels[cat] == 1]) & set(r.reviews.meta.loc[allowed, "source_id"])
        for m in modes:
            res = r.reviews.search(q, k, allowed, mode=m)
            hits = res["source_id"].isin(relevant).sum()
            ablation[m].append({"p_at_k": hits / k, "hit_at_k": float(hits > 0), "recall_capped": hits / min(k, len(relevant))})
        ret = r.search_reviews(q, game_id=gid, top_k=k)
        ans = answer(q, ret)
        sents = _sentences(ans["answer"])
        allowed_ids = {f"R:{i}" for i in ret["results"]["source_id"]}
        cit = validate_citations(ans["answer"], allowed_ids)
        grounded = sum(bool(validate_citations(x, allowed_ids)["valid"]) for x in sents) / max(1, len(sents))
        hits = ret["results"]["source_id"].isin(relevant).sum()
        rows.append({"type": "review", "question": q, "expected": f"{gid}/{cat}", "answered": ans["mode"] != "no_answer",
                     "should_answer": True, "p_at_k": hits / k, "hit_at_k": bool(hits > 0),
                     "citation_correctness": cit["citation_precision"], "groundedness": grounded,
                     "answer_relevance": CATEGORY_LABELS[cat] in ans["answer"], "mode": ans["mode"]})
    for q, doc_id in DOC_QUESTIONS:
        d = r.search_docs(q, top_k=3)
        ids = d["results"]["source_id"].tolist()
        rows.append({"type": "doc", "question": q, "expected": doc_id, "answered": d["answerable"], "should_answer": True,
                     "hit_at_k": doc_id in ids, "rank": ids.index(doc_id) + 1 if doc_id in ids else None})
    for q in UNANSWERABLE:
        ret = r.search_reviews(q, top_k=k)
        ans = answer(q, ret)
        rows.append({"type": "unanswerable", "question": q, "expected": "no answer", "answered": ans["mode"] != "no_answer",
                     "should_answer": False, "best_score": round(ret["best_score"], 4)})
    df = pd.DataFrame(rows)
    rev, doc, una = df[df.type == "review"], df[df.type == "doc"], df[df.type == "unanswerable"]
    summary = {
        "questions": int(len(df)), "selected_mode": r.mode, "review_questions": int(len(rev)), "doc_questions": int(len(doc)),
        "unanswerable_questions": int(len(una)), "k": k, "embedder": r.reviews.manifest["embedder"],
        "retrieval_ablation": {m: {kk: round(float(pd.DataFrame(v)[kk].mean()), 3) for kk in ("p_at_k", "hit_at_k", "recall_capped")}
                               for m, v in ablation.items()},
        "review_precision_at_k": round(float(rev["p_at_k"].mean()), 3), "review_hit_at_k": round(float(rev["hit_at_k"].mean()), 3),
        "doc_hit_at_3": round(float(doc["hit_at_k"].mean()), 3),
        "citation_correctness": round(float(rev["citation_correctness"].dropna().mean()), 3),
        "groundedness": round(float(rev["groundedness"].mean()), 3),
        "unsupported_claim_rate": round(1 - float(rev["groundedness"].mean()), 3),
        "answer_relevance_proxy": round(float(rev["answer_relevance"].mean()), 3),
        "no_answer_accuracy": round(float((df["answered"] == df["should_answer"]).mean()), 3),
        "unanswerable_refused": round(float((~una["answered"]).mean()), 3),
        "answerable_answered": round(float(df[df.should_answer]["answered"].mean()), 3),
        "answer_mode": "extractive (no LLM) - LLM answers were not evaluated in this environment",
    }
    out = s.artifacts_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rag_eval_cases.csv", index=False)
    (out / "rag_eval.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
