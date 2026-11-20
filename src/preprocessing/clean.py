"""Review cleaning and data-quality reporting.

Principles: the original text is never modified (`review_text`); a normalised copy
(`text_norm`) is added for analytics. Rows are FLAGGED rather than silently dropped, and
the `include` column decides what enters analytics/NLP. Every rule is counted in the
data-quality report.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
STOPWORDS = {
    "english": {"the", "and", "is", "it", "to", "of", "a", "i", "this", "in", "game", "not", "for", "my", "with",
                "no", "so", "are", "on", "an", "even", "be", "you", "me", "was", "but", "very", "just", "all", "too",
                "have", "has", "every", "after", "at", "make", "makes", "players", "time", "fun", "only", "far"},
    "spanish": {"el", "la", "de", "que", "y", "es", "muy", "juego", "no", "se", "los", "con"},
    "german": {"der", "die", "das", "und", "ist", "nicht", "sind", "ein", "zu", "mit", "ständig"},
    "french": {"le", "la", "les", "de", "et", "est", "je", "pas", "trop", "ne", "un", "des"},
    "portuguese": {"o", "a", "de", "que", "e", "muito", "jogo", "recomendo", "não", "um", "bom"},
}


def normalise(text: str) -> str:
    text = URL_RE.sub(" ", str(text))
    return WS_RE.sub(" ", text).strip().lower()


def detect_language(text: str) -> str:
    """Tiny stop-word vote; used to check (not replace) the source language field."""
    tokens = re.findall(r"[a-zà-ÿß]+", text.lower())
    if len(tokens) < 3:
        return "unknown"
    scores = {lang: sum(t in sw for t in tokens) / len(tokens) for lang, sw in STOPWORDS.items()}
    lang, score = max(scores.items(), key=lambda kv: kv[1])
    if lang != "english" and scores["english"] > 0:
        return "english"  # shared short words ("no", "a") must not flip clear English text
    return lang if score >= 0.08 else "unknown"


def is_spam(text: str) -> bool:
    t = re.sub(r"\s", "", text)
    if len(t) < 20:
        return False
    longest_run = max(len(m.group(0)) for m in re.finditer(r"(.)\1*", t))
    return longest_run >= 20 or len(set(t)) / len(t) < 0.05


def clean_reviews(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    report: dict = {"raw_rows": int(len(df)), "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    dup_id = df["review_id"].duplicated(keep="first")
    report["duplicate_review_id_removed"] = int(dup_id.sum())
    df = df[~dup_id].copy()

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["text_norm"] = df["review_text"].map(normalise)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
    df["voted_up"] = df["voted_up"].map({True: True, False: False, "True": True, "False": False, "true": True, "false": False})

    flags = pd.DataFrame(index=df.index)
    flags["empty_text"] = df["text_norm"].str.len() == 0
    flags["invalid_created_at"] = df["created_at"].isna()
    flags["missing_recommendation"] = df["voted_up"].isna()
    flags["spam_text"] = df["review_text"].map(is_spam)
    bad_update = df["updated_at"].notna() & df["created_at"].notna() & (df["updated_at"] < df["created_at"])
    flags["update_before_creation"] = bad_update
    df.loc[bad_update, "updated_at"] = pd.NaT  # keep the review, drop the impossible update time
    neg_play = df["playtime_hours"] < 0
    flags["negative_playtime"] = neg_play
    df.loc[neg_play, "playtime_hours"] = np.nan

    df["detected_language"] = df["review_text"].map(detect_language)
    flags["non_english"] = df["language"].ne("english") | df["detected_language"].isin(["spanish", "german", "french", "portuguese"])
    df["text_hash"] = df.apply(lambda r: hashlib.sha1(f"{r['game_id']}|{r['text_norm']}".encode(), usedforsecurity=False).hexdigest(), axis=1)
    flags["duplicate_text_same_game"] = df["text_hash"].duplicated(keep="first") & ~flags["empty_text"]

    hard = ["empty_text", "invalid_created_at", "missing_recommendation", "spam_text", "non_english"]
    df["include"] = ~flags[hard].any(axis=1)
    df["exclusion_reason"] = flags[hard].apply(lambda r: ",".join(r.index[r]), axis=1)
    df["is_text_duplicate"] = flags["duplicate_text_same_game"]  # kept but flagged; common short reviews repeat
    df["char_len"] = df["review_text"].str.len()
    df["word_len"] = df["review_text"].str.split().str.len()

    report["flags"] = {k: int(v) for k, v in flags.sum().items()}
    report["included_rows"] = int(df["include"].sum())
    report["excluded_rows"] = int((~df["include"]).sum())
    inc = df[df["include"]]
    report["included_summary"] = {
        "games": int(inc["game_id"].nunique()),
        "date_min": str(inc["created_at"].min()), "date_max": str(inc["created_at"].max()),
        "positive_share": float(inc["voted_up"].mean()),
        "word_len_median": float(inc["word_len"].median()), "word_len_p95": float(inc["word_len"].quantile(0.95)),
        "data_sources": sorted(inc["data_source"].unique().tolist()),
    }
    df["voted_up"] = df["voted_up"].astype("boolean")
    return df.reset_index(drop=True), report


def run(raw: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean, report = clean_reviews(raw)
    clean.to_parquet(out_dir / "reviews_clean.parquet", index=False)
    (out_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    return report
