"""Read-only analytics over a DuckDB database built by the pipeline.

* The pipeline writes artifacts/gameops.duckdb once; the API, dashboard and agent open it
  with read_only=True, so no caller can modify data.
* Every query is a fixed SQL string with bound parameters. Inputs are validated first
  (game ids against a whitelist, ISO dates, enumerated frequencies).
* Sentiment metrics use the SOURCE recommendation (voted_up). Complaint metrics use the
  model-predicted categories (see src/nlp/classifiers.py). Shares describe the collected
  reviews only - not all players of a game.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from src.config import CATEGORIES, CATEGORY_LABELS, get_settings

FREQS = {"day": "day", "week": "week", "month": "month"}
SMALL_SAMPLE = 30


class InvalidQuery(ValueError):
    """Raised for invalid user/tool arguments (mapped to HTTP 422 / tool error)."""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float | None:
    if min(n1, n2) == 0:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    z = (k2 / n2 - k1 / n1) / se
    return round(math.erfc(abs(z) / math.sqrt(2)), 6)


def build_database(clean: pd.DataFrame, predictions: pd.DataFrame, notes: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    inc = clean[clean["include"]].drop(columns=["include", "exclusion_reason"]).merge(predictions, on="review_id", how="left")
    inc["created_at"] = inc["created_at"].dt.tz_convert(None)
    inc["updated_at"] = inc["updated_at"].dt.tz_convert(None)
    inc["voted_up"] = inc["voted_up"].astype(bool)
    con.register("inc", inc)
    con.execute("CREATE TABLE reviews AS SELECT * FROM inc")
    notes_df = pd.DataFrame([{**n, "changes": " | ".join(n["changes"]), "release_date": pd.Timestamp(n["release_date"])} for n in notes])
    con.register("nd", notes_df)
    con.execute("CREATE TABLE update_notes AS SELECT * FROM nd")
    con.execute("CREATE INDEX ix_reviews_game_time ON reviews(game_id, created_at)")
    con.close()
    tmp.replace(path)


@dataclass
class Window:
    game_id: str | None
    start: date
    end: date


class AnalyticsStore:
    def __init__(self, db_path: Path | None = None):
        self.path = db_path or get_settings().artifacts_dir / "gameops.duckdb"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} missing. Run: python scripts/build_pipeline.py")
        self._local = threading.local()
        con = self._con()
        self.games = con.execute("SELECT game_id, ANY_VALUE(game_name) FROM reviews GROUP BY 1 ORDER BY 1").fetchall()
        self.game_ids = {g for g, _ in self.games}
        lo, hi = con.execute("SELECT MIN(created_at)::DATE, MAX(created_at)::DATE FROM reviews").fetchone()
        self.min_date, self.max_date = lo, hi

    def _con(self) -> duckdb.DuckDBPyConnection:
        if not hasattr(self._local, "con"):
            self._local.con = duckdb.connect(str(self.path), read_only=True)
        return self._local.con

    def q(self, sql: str, params: list) -> pd.DataFrame:
        return self._con().execute(sql, params).df()

    # ------------------------------------------------------------ validation
    def window(self, game_id: str | None = None, start: str | date | None = None, end: str | date | None = None) -> Window:
        if game_id is not None and game_id not in self.game_ids:
            raise InvalidQuery(f"Unknown game_id '{game_id}'. Valid: {sorted(self.game_ids)}")

        def parse(v, default):
            if v is None or v == "":
                return default
            if isinstance(v, (date, datetime)):
                return v if isinstance(v, date) and not isinstance(v, datetime) else v.date()
            try:
                return date.fromisoformat(str(v)[:10])
            except ValueError as exc:
                raise InvalidQuery(f"Invalid date '{v}' (use YYYY-MM-DD)") from exc
        s, e = parse(start, self.min_date), parse(end, self.max_date)
        if s > e:
            raise InvalidQuery(f"start {s} is after end {e}")
        return Window(game_id, s, e)

    @staticmethod
    def _where(w: Window) -> tuple[str, list]:
        sql = "created_at >= ? AND created_at < ? + INTERVAL 1 DAY"
        params: list = [w.start, w.end]
        if w.game_id:
            sql += " AND game_id = ?"
            params.append(w.game_id)
        return sql, params

    # ------------------------------------------------------------ metrics
    def overview(self, game_id=None, start=None, end=None) -> dict:
        w = self.window(game_id, start, end)
        where, p = self._where(w)
        r = self.q(f"""SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN voted_up THEN 1 ELSE 0 END), 0) pos, COUNT(DISTINCT game_id) games,
                       MIN(created_at) first_review, MAX(created_at) last_review FROM reviews WHERE {where}""", p).iloc[0]
        n, pos = int(r.n), int(r.pos or 0)
        names = dict(self.games)
        return {"game_id": w.game_id, "game_name": names.get(w.game_id), "start": str(w.start), "end": str(w.end), "total_reviews": n,
                "positive_reviews": pos, "negative_reviews": n - pos,
                "positive_pct": round(100 * pos / n, 2) if n else None, "negative_pct": round(100 * (n - pos) / n, 2) if n else None,
                "negative_share_95ci": wilson(n - pos, n), "games_analyzed": int(r.games),
                "first_review": str(r.first_review) if n else None, "last_review": str(r.last_review) if n else None,
                "small_sample": n < SMALL_SAMPLE}

    def sentiment_by_game(self, start=None, end=None) -> pd.DataFrame:
        w = self.window(None, start, end)
        where, p = self._where(w)
        df = self.q(f"""SELECT game_id, ANY_VALUE(game_name) game_name, COUNT(*) reviews,
                        AVG(CASE WHEN voted_up THEN 0 ELSE 1 END) negative_share,
                        AVG(CASE WHEN model_sentiment = 'negative' THEN 1 ELSE 0 END) model_negative_share
                        FROM reviews WHERE {where} GROUP BY game_id ORDER BY game_id""", p)
        return df

    def sentiment_timeseries(self, game_id=None, start=None, end=None, freq: str = "week") -> pd.DataFrame:
        if freq not in FREQS:
            raise InvalidQuery(f"freq must be one of {list(FREQS)}")
        w = self.window(game_id, start, end)
        where, p = self._where(w)
        df = self.q(f"""SELECT date_trunc('{FREQS[freq]}', created_at)::DATE AS period, COUNT(*) reviews,
                        SUM(CASE WHEN voted_up THEN 1 ELSE 0 END) positive,
                        SUM(CASE WHEN voted_up THEN 0 ELSE 1 END) negative
                        FROM reviews WHERE {where} GROUP BY 1 ORDER BY 1""", p)
        df["negative_share"] = df["negative"] / df["reviews"]
        ci = [wilson(int(k), int(n)) for k, n in zip(df["negative"], df["reviews"], strict=False)]
        df["ci_low"], df["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
        return df

    def complaints(self, game_id=None, start=None, end=None, scope: str = "negative") -> dict:
        if scope not in ("negative", "all"):
            raise InvalidQuery("scope must be 'negative' or 'all'")
        w = self.window(game_id, start, end)
        where, p = self._where(w)
        if scope == "negative":
            where += " AND NOT voted_up"
        sums = ", ".join(f"COALESCE(SUM(CASE WHEN {c} THEN 1 ELSE 0 END), 0) AS {c}" for c in CATEGORIES)
        r = self.q(f"SELECT COUNT(*) n, {sums} FROM reviews WHERE {where}", p).iloc[0]
        n = int(r.n)
        cats = [{"category": c, "label": CATEGORY_LABELS[c], "reviews": int(r[c] or 0),
                 "share": round(int(r[c] or 0) / n, 4) if n else None, "share_95ci": wilson(int(r[c] or 0), n)}
                for c in CATEGORIES]
        cats.sort(key=lambda d: d["reviews"], reverse=True)
        return {"game_id": w.game_id, "start": str(w.start), "end": str(w.end), "scope": scope, "base_reviews": n,
                "categories": cats, "small_sample": n < SMALL_SAMPLE,
                "note": "Categories are model predictions; a review can have several."}

    def complaint_trends(self, game_id=None, start=None, end=None, freq: str = "week", scope: str = "negative") -> pd.DataFrame:
        if freq not in FREQS:
            raise InvalidQuery(f"freq must be one of {list(FREQS)}")
        w = self.window(game_id, start, end)
        where, p = self._where(w)
        if scope == "negative":
            where += " AND NOT voted_up"
        sums = ", ".join(f"SUM(CASE WHEN {c} THEN 1 ELSE 0 END) AS {c}" for c in CATEGORIES)
        df = self.q(f"""SELECT date_trunc('{FREQS[freq]}', created_at)::DATE AS period, COUNT(*) reviews, {sums}
                        FROM reviews WHERE {where} GROUP BY 1 ORDER BY 1""", p)
        long = df.melt(id_vars=["period", "reviews"], value_vars=CATEGORIES, var_name="category", value_name="count")
        long["share"] = long["count"] / long["reviews"]
        return long

    def compare_periods(self, game_id, a_start, a_end, b_start, b_end) -> dict:
        wa, wb = self.window(game_id, a_start, a_end), self.window(game_id, b_start, b_end)
        out = {}
        for name, w in (("period_a", wa), ("period_b", wb)):
            ov = self.overview(w.game_id, w.start, w.end)
            cp = self.complaints(w.game_id, w.start, w.end, scope="all")
            out[name] = {"start": str(w.start), "end": str(w.end), "reviews": ov["total_reviews"],
                         "negative": ov["negative_reviews"], "negative_pct": ov["negative_pct"],
                         "categories": {c["category"]: c["reviews"] for c in cp["categories"]}}
        a, b = out["period_a"], out["period_b"]
        changes = []
        for c in CATEGORIES:
            ka, kb = a["categories"][c], b["categories"][c]
            sa = ka / a["reviews"] if a["reviews"] else None
            sb = kb / b["reviews"] if b["reviews"] else None
            changes.append({"category": c, "label": CATEGORY_LABELS[c], "share_a": sa, "share_b": sb,
                            "change_pct_points": round(100 * (sb - sa), 2) if sa is not None and sb is not None else None,
                            "p_value": two_prop_p(ka, a["reviews"], kb, b["reviews"])})
        changes.sort(key=lambda d: -(d["change_pct_points"] or 0))
        out.update({"game_id": game_id, "negative_share_change_pct_points":
                    round(b["negative_pct"] - a["negative_pct"], 2) if a["negative_pct"] is not None and b["negative_pct"] is not None else None,
                    "negative_share_p_value": two_prop_p(a["negative"], a["reviews"], b["negative"], b["reviews"]),
                    "category_changes": changes,
                    "small_sample": min(a["reviews"], b["reviews"]) < SMALL_SAMPLE,
                    "caution": "Differences between periods are associations; they do not prove what caused them."})
        return out

    def emerging_issues(self, game_id=None, days: int = 28, end=None, alpha: float = 0.05) -> list[dict]:
        w = self.window(game_id, None, end)
        b_end = w.end
        b_start = b_end - pd.Timedelta(days=days - 1)
        a_end, a_start = b_start - pd.Timedelta(days=1), b_start - pd.Timedelta(days=days)
        cmp = self.compare_periods(game_id, a_start, a_end, b_start, b_end)
        return [{**c, "period_a": cmp["period_a"]["start"] + ".." + cmp["period_a"]["end"],
                 "period_b": cmp["period_b"]["start"] + ".." + cmp["period_b"]["end"]}
                for c in cmp["category_changes"]
                if c["change_pct_points"] and c["change_pct_points"] > 0 and c["p_value"] is not None and c["p_value"] < alpha]

    def representative_reviews(self, category: str, game_id=None, start=None, end=None, limit: int = 5,
                               negative_only: bool = True) -> pd.DataFrame:
        if category not in CATEGORIES:
            raise InvalidQuery(f"category must be one of {CATEGORIES}")
        limit = max(1, min(int(limit), 50))
        w = self.window(game_id, start, end)
        where, p = self._where(w)
        if negative_only:
            where += " AND NOT voted_up"
        return self.q(f"""SELECT review_id, game_name, created_at, voted_up, helpful_votes, review_text FROM (
                            SELECT *, ROW_NUMBER() OVER (PARTITION BY text_hash ORDER BY helpful_votes DESC, review_id) rn
                            FROM reviews WHERE {where} AND {category}) t
                          WHERE rn = 1 ORDER BY helpful_votes DESC, created_at DESC LIMIT {limit}""", p)

    def update_notes(self, game_id=None) -> pd.DataFrame:
        if game_id is not None and game_id not in self.game_ids:
            raise InvalidQuery(f"Unknown game_id '{game_id}'")
        if game_id:
            return self.q("SELECT * FROM update_notes WHERE game_id = ? ORDER BY release_date", [game_id])
        return self.q("SELECT * FROM update_notes ORDER BY release_date", [])

    def reviews_frame(self, game_id=None, start=None, end=None, columns: tuple = ("review_id", "game_id", "game_name",
                      "created_at", "voted_up", "helpful_votes", "playtime_hours", "review_text")) -> pd.DataFrame:
        w = self.window(game_id, start, end)
        where, p = self._where(w)
        allowed = {"review_id", "game_id", "game_name", "created_at", "updated_at", "voted_up", "helpful_votes",
                   "playtime_hours", "review_text", "platform", "model_sentiment", "model_p_negative", *CATEGORIES}
        bad = [c for c in columns if c not in allowed]
        if bad:
            raise InvalidQuery(f"Unknown columns {bad}")
        cols = ", ".join(columns)
        return self.q(f"SELECT {cols} FROM reviews WHERE {where} ORDER BY created_at", p)
