"""Choose the default review retrieval mode on the DEV question set (never the eval set)."""

from __future__ import annotations

import json

import pandas as pd

from src.config import get_settings
from src.evaluation.rag_questions import DEV_PAIRS, DEV_PHRASES
from src.rag.retriever import Retriever

MODES = ("dense", "lexical", "hybrid", "sum")


def select(k: int = 5) -> dict:
    s = get_settings()
    r = Retriever()
    labels = pd.read_parquet(s.raw_dir / "labels_synthetic.parquet").set_index("review_id")
    games = dict(r.reviews.meta.groupby("game_id")["game_name"].first())
    scores = {m: [] for m in MODES}
    for gid, cat in DEV_PAIRS:
        q = f"Show {games[gid]} reviews about {DEV_PHRASES[cat]}"
        allowed = r._allowed(gid, None, None, None)
        relevant = set(labels.index[labels[cat] == 1])
        for m in MODES:
            res = r.reviews.search(q, k, allowed, mode=m)
            scores[m].append(res["source_id"].isin(relevant).sum() / k)  # empty result counts as 0
    summary = {m: round(float(sum(v) / len(v)), 3) for m, v in scores.items()}
    best = max(summary, key=summary.get)
    out = {"dev_precision_at_k": summary, "selected_mode": best, "dev_questions": len(DEV_PAIRS)}
    (s.artifacts_dir / "index" / "retrieval_mode.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(select(), indent=2))
