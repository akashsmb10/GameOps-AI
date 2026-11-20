import pandas as pd

from src.preprocessing.clean import clean_reviews, detect_language, is_spam


def raw(**over):
    base = dict(review_id="R1", game_id="G1", game_name="Game", review_text="The servers crash a lot.", voted_up=False,
                created_at="2026-05-01T10:00:00Z", updated_at="2026-05-02T10:00:00Z", platform="windows",
                playtime_hours=3.0, helpful_votes=0, language="english", data_source="synthetic_demo")
    base.update(over)
    return base


def test_cleaning_flags_each_defect_and_keeps_original_text():
    df = pd.DataFrame([raw(), raw(),  # duplicate id
                       raw(review_id="R2", review_text="   "),
                       raw(review_id="R3", created_at="not-a-date"),
                       raw(review_id="R4", review_text="a" * 80),
                       raw(review_id="R5", review_text="El juego se cierra constantemente", language="spanish"),
                       raw(review_id="R6", updated_at="2026-01-01T00:00:00Z", playtime_hours=-2.0),
                       raw(review_id="R7", review_text="GREAT Game!!  http://x.y  ")])
    out, rep = clean_reviews(df)
    assert rep["duplicate_review_id_removed"] == 1
    f = rep["flags"]
    assert f["empty_text"] == 1 and f["invalid_created_at"] == 1 and f["spam_text"] == 1 and f["non_english"] == 1
    assert f["update_before_creation"] == 1 and f["negative_playtime"] == 1
    r6 = out.set_index("review_id").loc["R6"]
    assert r6["include"] and pd.isna(r6["updated_at"]) and pd.isna(r6["playtime_hours"])
    r7 = out.set_index("review_id").loc["R7"]
    assert r7["review_text"] == "GREAT Game!!  http://x.y  " and r7["text_norm"] == "great game!!"
    assert set(out[out["include"]]["review_id"]) == {"R1", "R6", "R7"}


def test_language_detector_does_not_flip_english_with_shared_words():
    assert detect_language("Ratings make no sense, low rated players outperform stars.") == "english"
    assert detect_language("Die Server sind ständig offline und das ist nicht gut") == "german"
    assert not is_spam("Short text") and is_spam("x" * 30)
