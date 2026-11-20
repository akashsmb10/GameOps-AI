"""Check the no-answer gate on calibration probes and record it with the index.

The probes below are deliberately DIFFERENT from the RAG evaluation questions in
src/evaluation/rag_questions.py, so the evaluation is not used to tune the system.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.nlp.embeddings import get_embedder

from .index import VectorIndex

IN_DOMAIN = ["players complaining that it crashes after loading", "frame drops and stutter on good hardware",
             "kicked from server in the middle of a match", "too expensive cosmetics and paid boosts",
             "waiting a long time in the matchmaking queue", "menus are hard to use", "ran out of things to do",
             "support team ignored my request", "one weapon is much stronger than the others", "really fun with friends"]
OFF_DOMAIN = ["what is the capital of france", "recipe for chicken biryani", "share price of a semiconductor company",
              "how to change a car tyre", "weather forecast for tomorrow", "python list comprehension syntax",
              "who won the 2010 football world cup final", "best home loan interest rates", "symptoms of seasonal flu",
              "translate good morning into japanese"]


def calibrate(index_dir: Path) -> dict:
    manifest = json.loads((index_dir / "reviews_manifest.json").read_text())
    family = manifest["embedder"].split(":")[0]
    idx = VectorIndex.load(index_dir, "reviews", get_embedder(family if family != "tfidf-svd" else "tfidf"))
    # Gate = query-term coverage (see VectorIndex.coverage). Score thresholds were tried first
    # and rejected: with averaged word vectors, short on-topic queries score below off-topic
    # ones, so no cosine threshold separates them (documented in docs/RAG_EVALUATION.md).
    pos = np.array([idx.coverage(q) for q in IN_DOMAIN])
    neg = np.array([idx.coverage(q) for q in OFF_DOMAIN])
    MIN_COVERAGE = 0.5  # fixed a priori ("at least half the content words are known"), not tuned
    out = {"min_coverage": MIN_COVERAGE, "min_score": 0.0, "gate": "query-term coverage >= 0.5",
           "in_domain_accepted": float((pos >= MIN_COVERAGE).mean()), "off_domain_rejected": float((neg < MIN_COVERAGE).mean()),
           "in_domain_coverage": pos.round(3).tolist(), "off_domain_coverage": neg.round(3).tolist(),
           "embedder": manifest["embedder"]}
    (index_dir / "retrieval_calibration.json").write_text(json.dumps(out, indent=2))
    return out
