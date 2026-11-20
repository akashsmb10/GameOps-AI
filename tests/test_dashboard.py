from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

PAGES = ["overview", "sentiment", "complaints", "updates", "analyst", "evaluation", "playground"]


def _open(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py")
        at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    assert not _open(page).exception


def test_game_filter_changes_the_numbers():
    at = _open("overview")
    before = [m.value for m in at.markdown if "class=\"gp-kpi\"" in m.value][0]
    at.selectbox(key="f_game").select_index(1).run()
    after = [m.value for m in at.markdown if "class=\"gp-kpi\"" in m.value][0]
    assert not at.exception and before != after
