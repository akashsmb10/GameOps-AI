"""Real-data connector: Steam's public store review endpoint and Steam News announcements.

Endpoints (public, no API key):
  https://store.steampowered.com/appreviews/<appid>?json=1&cursor=...   (reviews, 100/page)
  https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=...   (official announcements)

Usage rules followed here:
* Polite rate limit (default 1.5 s between pages) and a hard page cap.
* No reviewer identifiers are stored: the author's steamid is dropped on ingestion.
* Review text is user-generated content owned by its authors under Valve's terms. Collected
  data is written to data/raw/steam/ which is git-ignored - do NOT commit or redistribute it.
* Steam does not report an operating system per review; we keep only the documented
  `primarily_steam_deck` flag, so `platform` is "steam_deck" or "steam_pc_or_other".

This connector could not be executed in the build environment (network blocked); it is
unit-tested against a hand-written fixture in the documented response format
(tests/fixtures/steam_appreviews_sample.json).
"""

from __future__ import annotations

import argparse
import json
import time
from urllib.parse import quote

import pandas as pd
import requests

from src.config import get_settings

REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"


def parse_reviews(payload: dict, appid: int, game_name: str) -> pd.DataFrame:
    if not payload.get("success"):
        raise ValueError("Steam response reports success != 1")
    rows = []
    for r in payload.get("reviews", []):
        author = r.get("author", {})
        rows.append({
            "review_id": f"steam-{r['recommendationid']}", "game_id": f"steam:{appid}", "game_name": game_name,
            "review_text": r.get("review", ""), "voted_up": bool(r.get("voted_up")),
            "created_at": pd.to_datetime(r.get("timestamp_created"), unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated_at": pd.to_datetime(r.get("timestamp_updated"), unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": "steam_deck" if r.get("primarily_steam_deck") else "steam_pc_or_other",
            "playtime_hours": round(author.get("playtime_at_review", 0) / 60, 1) if author.get("playtime_at_review") is not None else None,
            "helpful_votes": int(r.get("votes_up", 0)), "language": r.get("language", "unknown"),
            "data_source": "steam_store_api"})
    return pd.DataFrame(rows)


def fetch_reviews(appid: int, game_name: str, max_reviews: int = 2000, language: str = "english",
                  pause_s: float = 1.5, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or requests.Session()
    cursor, frames, seen = "*", [], 0
    while seen < max_reviews:
        params = {"json": 1, "filter": "recent", "language": language, "num_per_page": 100,
                  "purchase_type": "all", "review_type": "all"}
        url = REVIEWS_URL.format(appid=appid) + f"?cursor={quote(cursor)}"
        resp = sess.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        page = parse_reviews(payload, appid, game_name)
        if page.empty:
            break
        frames.append(page)
        seen += len(page)
        new_cursor = payload.get("cursor")
        if not new_cursor or new_cursor == cursor:
            break
        cursor = new_cursor
        time.sleep(pause_s)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.head(max_reviews)


def parse_news(payload: dict, appid: int, game_name: str) -> list[dict]:
    items = payload.get("appnews", {}).get("newsitems", [])
    return [{"doc_id": f"steam-news-{n['gid']}", "game_id": f"steam:{appid}", "game_name": game_name,
             "version": n.get("title", ""), "release_date": pd.to_datetime(n["date"], unit="s").strftime("%Y-%m-%d"),
             "changes": [n.get("contents", "")], "url": n.get("url"), "data_source": "steam_news_api"}
            for n in items if n.get("feedname") == "steam_community_announcements"]


def fetch_news(appid: int, game_name: str, count: int = 50, session: requests.Session | None = None) -> list[dict]:
    sess = session or requests.Session()
    resp = sess.get(NEWS_URL, params={"appid": appid, "count": count, "maxlength": 3000,
                                      "feeds": "steam_community_announcements"}, timeout=30)
    resp.raise_for_status()
    return parse_news(resp.json(), appid, game_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Collect REAL Steam reviews locally (not for redistribution)")
    ap.add_argument("--appid", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-reviews", type=int, default=2000)
    a = ap.parse_args()
    out = get_settings().raw_dir / "steam"
    out.mkdir(parents=True, exist_ok=True)
    df = fetch_reviews(a.appid, a.name, a.max_reviews)
    df.to_csv(out / f"reviews_{a.appid}.csv", index=False)
    (out / f"news_{a.appid}.json").write_text(json.dumps(fetch_news(a.appid, a.name), indent=2))
    print(f"saved {len(df)} reviews to {out}")
