import hashlib
import json
from pathlib import Path

import pandas as pd

from src.ingestion import steam
from src.ingestion.demo_generator import generate
from src.ingestion.loader import RAW_COLUMNS, load_raw_reviews

FIX = Path(__file__).parent / "fixtures" / "steam_appreviews_sample.json"


def _h(df):
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def test_demo_generator_is_reproducible():
    a, la, _ = generate(400, seed=3)
    b, lb, _ = generate(400, seed=3)
    c, _, _ = generate(400, seed=4)
    assert _h(a) == _h(b) and _h(la) == _h(lb)
    assert _h(a) != _h(c)


def test_demo_data_is_labelled_synthetic_and_has_schema():
    df = load_raw_reviews()
    assert list(df.columns) == RAW_COLUMNS
    assert set(df["data_source"]) == {"synthetic_demo"}


def test_steam_parser_maps_documented_fields_and_drops_author_id():
    payload = json.loads(FIX.read_text())
    df = steam.parse_reviews(payload, 123, "Fixture Game")
    assert len(df) == 2 and set(df.columns) == set(RAW_COLUMNS)
    assert df.loc[0, "review_id"] == "steam-900000001" and df.loc[0, "playtime_hours"] == 8.0
    assert df.loc[1, "platform"] == "steam_deck" and not df.loc[0, "voted_up"]
    assert "76500000000000001" not in df.to_csv()  # reviewer steamid is not stored


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, pages):
        self.pages, self.urls = pages, []

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        return FakeResp(self.pages.pop(0) if self.pages else {"success": 1, "reviews": [], "cursor": "end"})


def test_steam_fetch_paginates_with_cursor_and_stops():
    page = json.loads(FIX.read_text())
    sess = FakeSession([page, {**page, "cursor": "AoJ4fixturecursor"}])
    df = steam.fetch_reviews(1, "G", max_reviews=10, pause_s=0, session=sess)
    assert len(df) == 4 and "cursor=%2A" in sess.urls[0] and "AoJ4fixturecursor" in sess.urls[1]
