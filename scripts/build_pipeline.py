"""End-to-end build: clean -> NLP models -> predictions -> DuckDB -> FAISS indexes.

    python scripts/build_pipeline.py            # demo dataset (default)
    GAMEOPS_DATASET=steam python scripts/build_pipeline.py
"""
import json
import logging
import time
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import pandas as pd

from src.analytics.store import build_database
from src.config import get_settings
from src.ingestion.demo_generator import write as write_demo
from src.ingestion.loader import load_raw_reviews, load_update_notes
from src.nlp.classifiers import predict_all, train_and_evaluate
from src.nlp.embeddings import get_embedder
from src.preprocessing.clean import run as clean_run
from src.rag.calibrate import calibrate
from src.rag.index import VectorIndex, doc_chunks, review_chunks
from src.rag.select_mode import select as select_mode


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    s = get_settings()
    t0 = time.perf_counter()
    if s.dataset == "demo" and not (s.raw_dir / "demo_reviews.csv").exists():
        write_demo(s.raw_dir)
    dq = clean_run(load_raw_reviews(), s.processed_dir)
    clean = pd.read_parquet(s.processed_dir / "reviews_clean.parquet")
    labels = pd.read_parquet(s.raw_dir / "labels_synthetic.parquet") if s.dataset == "demo" else None
    embedder = get_embedder(s.embedder, s.st_model)
    nlp = train_and_evaluate(clean, labels, embedder, s.artifacts_dir / "nlp")
    preds = predict_all(clean, nlp["selected_model"], nlp["sentiment_model"], nlp["sentiment_threshold"])
    preds.to_parquet(s.processed_dir / "review_predictions.parquet", index=False)
    notes = load_update_notes()
    build_database(clean, preds, notes, s.artifacts_dir / "gameops.duckdb")

    inc = clean[clean["include"]].merge(preds, on="review_id")
    idx_dir = s.artifacts_dir / "index"
    rev_idx = VectorIndex.build(review_chunks(inc), embedder, "reviews")
    rev_idx.save(idx_dir)
    VectorIndex.build(doc_chunks(notes, s.knowledge_dir), rev_idx.embedder, "docs").save(idx_dir)
    mode = select_mode() if s.dataset == "demo" else {"selected_mode": "hybrid"}  # dev labels exist only for demo
    cal = calibrate(idx_dir)
    manifest = {"dataset": s.dataset, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "included_reviews": dq["included_rows"], "embedder": rev_idx.manifest["embedder"],
                "review_vectors": rev_idx.manifest["vectors"], "complaint_model": nlp["metrics"]["selection"]["selected"],
                "label_source": nlp["metrics"]["label_source"], "retrieval_mode": mode["selected_mode"],
                "no_answer_gate": cal["gate"],
                "seconds": round(time.perf_counter() - t0, 1)}
    (s.artifacts_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
