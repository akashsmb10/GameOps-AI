import duckdb
import pandas as pd
import pytest

from src.agents.tools import store
from src.analytics.store import InvalidQuery, two_prop_p, wilson
from src.config import CATEGORIES, get_settings


@pytest.fixture(scope="module")
def frame():
    clean = pd.read_parquet(get_settings().processed_dir / "reviews_clean.parquet")
    preds = pd.read_parquet(get_settings().processed_dir / "review_predictions.parquet")
    return clean[clean["include"]].merge(preds, on="review_id")


def test_overview_matches_independent_pandas_count(frame):
    st = store()
    gid = sorted(frame["game_id"].unique())[0]
    sub = frame[(frame.game_id == gid) & (frame.created_at.dt.date >= pd.Timestamp("2026-06-01").date())
                & (frame.created_at.dt.date <= pd.Timestamp("2026-06-30").date())]
    ov = st.overview(gid, "2026-06-01", "2026-06-30")
    assert ov["total_reviews"] == len(sub)
    assert ov["negative_reviews"] == int((~sub["voted_up"].astype(bool)).sum())


def test_complaint_counts_match_predictions(frame):
    comp = store().complaints(None, None, None, scope="negative")
    neg = frame[~frame["voted_up"].astype(bool)]
    got = {c["category"]: c["reviews"] for c in comp["categories"]}
    assert comp["base_reviews"] == len(neg)
    assert all(got[c] == int(neg[c].sum()) for c in CATEGORIES)


def test_date_filter_is_inclusive_on_both_ends(frame):
    day = frame["created_at"].dt.date.min()
    assert store().overview(None, day, day)["total_reviews"] == int((frame["created_at"].dt.date == day).sum())


@pytest.mark.parametrize("args", [("G999", None, None), (None, "2026-02-30", None), (None, "2026-07-01", "2026-06-01"),
                                  ("G001' OR 1=1 --", None, None)])
def test_invalid_inputs_are_rejected(args):
    with pytest.raises(InvalidQuery):
        store().overview(*args)


def test_column_whitelist_blocks_sql_in_identifiers():
    with pytest.raises(InvalidQuery):
        store().reviews_frame(columns=("review_id", "1; DROP TABLE reviews"))
    with pytest.raises(InvalidQuery):
        store().representative_reviews("bugs OR 1=1")


def test_database_connection_is_read_only():
    con = store()._con()
    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM reviews")


def test_wilson_and_two_proportion_test():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson(0, 0) == (None, None)
    assert two_prop_p(10, 100, 30, 100) < 0.001 and two_prop_p(10, 100, 11, 100) > 0.5


def test_compare_periods_reports_changes_and_caution():
    out = store().compare_periods("G001", "2026-06-01", "2026-06-30", "2026-07-08", "2026-07-31")
    assert {"period_a", "period_b", "category_changes", "negative_share_p_value"} <= set(out)
    assert "do not prove" in out["caution"]
