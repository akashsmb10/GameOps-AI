"""Deterministic planner used when no LLM is available (clearly labelled in every response).

It extracts the game, time window and complaint categories from the question, recognises the
task type, and produces a list of tool calls. With an LLM, the model chooses the tools
instead (src/agents/graph.py); both paths use the same validated tools.

Relative dates ("last week", "this month") are anchored to the LATEST REVIEW in the dataset,
not to today's clock, because the data is historical.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.config import CATEGORIES
from src.nlp.rules import LEXICON

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
CATEGORY_WORDS = {c: set(LEXICON.get(c, [])) | {c} for c in CATEGORIES}
CATEGORY_WORDS["servers"] |= {"server", "connectivity", "network", "online"}
CATEGORY_WORDS["monetization"] |= {"monetisation", "pricing", "price", "purchases", "money"}
CATEGORY_WORDS["ux"] |= {"user experience", "ui", "controls"}


@dataclass
class Entities:
    game_ids: list[str] = field(default_factory=list)
    periods: list[tuple[date, date]] = field(default_factory=list)  # chronological
    categories: list[str] = field(default_factory=list)
    version: str | None = None
    notes: list[str] = field(default_factory=list)


def _month_range(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def extract(question: str, games: list[tuple[str, str]], anchor: date, patch_dates: dict) -> Entities:
    q = question.lower()
    e = Entities()
    for gid, name in games:
        words = [w for w in name.lower().split() if len(w) >= 5]
        if gid.lower() in q or name.lower() in q or any(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            e.game_ids.append(gid)
    for c, words in CATEGORY_WORDS.items():
        if c != "other" and any(re.search(rf"\b{re.escape(w.strip())}", q) for w in words if w.strip()):
            e.categories.append(c)
    iso = re.findall(r"\b(20\d\d-\d\d-\d\d)\b", q)
    m_ver = re.search(r"(?:update|patch|version|v)\s*(\d+(?:\.\d+)+)", q)
    if m_ver:
        e.version = m_ver.group(1)
    if len(iso) >= 4:
        d = [date.fromisoformat(x) for x in iso[:4]]
        e.periods = [(d[0], d[1]), (d[2], d[3])]
    elif len(iso) >= 2:
        e.periods = [(date.fromisoformat(iso[0]), date.fromisoformat(iso[1]))]
    elif e.version and len(e.game_ids) == 1 and (e.game_ids[0], e.version) in patch_dates:
        p = patch_dates[(e.game_ids[0], e.version)]
        e.periods = [(p - timedelta(days=21), p - timedelta(days=1)), (p, p + timedelta(days=20))]
        e.notes.append(f"Compared the 21 days before update {e.version} ({p}) with the 21 days from its release.")
    else:
        named = [MONTHS[m] for m in re.findall(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", q)]
        if named:
            e.periods = [_month_range(anchor.year, m) for m in named[:2]]
            e.periods.sort()
        elif re.search(r"\b(last|past|this) week\b", q):
            e.periods = [(anchor - timedelta(days=13), anchor - timedelta(days=7)), (anchor - timedelta(days=6), anchor)]
            e.notes.append(f"'Week' is anchored to the latest review date in the data ({anchor}).")
        elif re.search(r"\b(last|past|this) month\b", q):
            e.periods = [(anchor - timedelta(days=59), anchor - timedelta(days=30)), (anchor - timedelta(days=29), anchor)]
            e.notes.append(f"'Month' means the 30 days ending at the latest review date in the data ({anchor}).")
        else:
            m_days = re.search(r"\b(?:last|past) (\d{1,3}) days\b", q)
            if m_days:
                n = int(m_days.group(1))
                e.periods = [(anchor - timedelta(days=2 * n - 1), anchor - timedelta(days=n)), (anchor - timedelta(days=n - 1), anchor)]
    return e


INTENTS = {
    "compare": r"\b(why|increase|increased|rise|rose|spike|drop|dropped|decrease|decreased|fell|change|changed|compare|versus|vs|before|after|trend|trends)\b",
    "complaints": r"\b(complain\w*|complaints?|issues?|problems?|most common|top|biggest|main|investigate|prioriti[sz]e|first)\b",
    "evidence": r"\b(summari[sz]e|examples?|what do players say|what are players saying|show|quotes?|reviews? about|latest negative)\b",
    "docs": r"\b(patch notes?|update notes?|release notes?|what changed|changelog|definition|defined|calculated|how is|metric|limitations?)\b",
    "quality": r"\b(data quality|excluded|cleaning|duplicates?|provenance|how many reviews were|dataset|model version|accuracy|f1)\b",
}


def plan(question: str, e: Entities, anchor: date) -> list[tuple[str, dict]]:
    q = question.lower()
    hit = {k: bool(re.search(p, q)) for k, p in INTENTS.items()}
    calls: list[tuple[str, dict]] = []
    games = e.game_ids or [None]
    if hit["quality"] and not hit["compare"]:
        return [("get_data_quality_report", {})]
    feedback_words = re.search(r"\b(reviews?|negative|positive|sentiment|complain\w*|players? (say|think|feel)|unhappy)\b", q)
    if re.search(r"\bwhat changed in\b|\b(patch|update|release) notes?\b|\bchangelog\b", q) and not feedback_words:
        return [("retrieve_game_documentation", {"query": question[:300], "game_id": games[0]})]
    if hit["docs"] and not hit["compare"]:
        calls.append(("retrieve_game_documentation", {"query": question[:300], "game_id": games[0]}))
        if not hit["complaints"] and not hit["evidence"]:
            return calls
    compare_games = len(e.game_ids) >= 2 and re.search(r"\b(compare|versus|vs|between)\b", q)
    if compare_games:
        start, end = (e.periods[-1] if e.periods else (None, None))
        for g in e.game_ids[:3]:
            calls.append(("analyze_complaints", {"game_id": g, "start_date": _s(start), "end_date": _s(end), "scope": "negative"}))
        return calls
    if hit["compare"] and (len(e.periods) == 2 or e.version):
        (a0, a1), (b0, b1) = e.periods
        for g in games[:1]:
            calls.append(("compare_feedback_trends", {"game_id": g, "period_a_start": _s(a0), "period_a_end": _s(a1),
                                                      "period_b_start": _s(b0), "period_b_end": _s(b1)}))
            calls.append(("analyze_complaints", {"game_id": g, "start_date": _s(b0), "end_date": _s(b1), "scope": "negative"}))
            topic = " ".join(e.categories) or "problems complaints"
            calls.append(("search_player_reviews", {"query": f"{topic} negative feedback", "game_id": g, "start_date": _s(b0),
                                                    "end_date": _s(b1), "sentiment": "negative", "top_k": 4}))
            if g:
                calls.append(("retrieve_game_documentation", {"query": f"update patch notes {e.version or ''}".strip(), "game_id": g}))
        return calls
    if hit["compare"] and not e.periods:
        # trend question without explicit periods: last 28 days vs the 28 days before
        b1 = anchor
        b0, a1 = b1 - timedelta(days=27), b1 - timedelta(days=28)
        a0 = a1 - timedelta(days=27)
        e.notes.append(f"No period was given, so I compared {b0}..{b1} with {a0}..{a1} (latest data first).")
        e.periods = [(a0, a1), (b0, b1)]
        return plan(question, e, anchor)
    start, end = (e.periods[-1] if e.periods else (None, None))
    if hit["complaints"] or not (hit["evidence"] or hit["docs"]):
        for g in games[:3]:
            calls.append(("analyze_complaints", {"game_id": g, "start_date": _s(start), "end_date": _s(end), "scope": "negative"}))
    if hit["evidence"] or e.categories or not calls:
        topic = " ".join(e.categories) if e.categories else question[:200]
        sentiment = "negative" if re.search(r"negative|complain|problem|issue|unhappy|bad", q) else "any"
        calls.append(("search_player_reviews", {"query": topic, "game_id": games[0], "start_date": _s(start), "end_date": _s(end),
                                                "sentiment": sentiment, "top_k": 5}))
    return calls


def _s(d) -> str | None:
    return d.isoformat() if d else None
