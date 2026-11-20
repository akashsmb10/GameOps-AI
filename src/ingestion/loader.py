"""Load raw reviews and update documents for the configured dataset (demo or steam)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import get_settings

RAW_COLUMNS = ["review_id", "game_id", "game_name", "review_text", "voted_up", "created_at", "updated_at",
               "platform", "playtime_hours", "helpful_votes", "language", "data_source"]


def load_raw_reviews(dataset: str | None = None, raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or get_settings().raw_dir
    dataset = dataset or get_settings().dataset
    if dataset == "demo":
        path = raw_dir / "demo_reviews.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/generate_demo_data.py")
        df = pd.read_csv(path, dtype={"review_id": str, "created_at": str, "updated_at": str})
    elif dataset == "steam":
        files = sorted((raw_dir / "steam").glob("reviews_*.csv"))
        if not files:
            raise FileNotFoundError("No Steam files. Run: python -m src.ingestion.steam --appid <id> --name <game>")
        df = pd.concat([pd.read_csv(f, dtype={"review_id": str}) for f in files], ignore_index=True)
    else:
        raise ValueError(f"Unknown dataset '{dataset}' (use demo or steam)")
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw reviews missing columns: {sorted(missing)}")
    return df[RAW_COLUMNS]


def load_update_notes(dataset: str | None = None, raw_dir: Path | None = None) -> list[dict]:
    raw_dir = raw_dir or get_settings().raw_dir
    dataset = dataset or get_settings().dataset
    if dataset == "demo":
        return json.loads((raw_dir / "demo_patch_notes.json").read_text())
    notes: list[dict] = []
    for f in sorted((raw_dir / "steam").glob("news_*.json")):
        notes.extend(json.loads(f.read_text()))
    return notes
