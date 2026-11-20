"""Shared dashboard pieces: theme, Plotly template, data access, filters and UI components.

Data access goes through the same read-only services as the API (src/analytics, src/rag,
src/agents) - documented data-access layer, no screenshots or hard-coded numbers.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CATEGORIES, CATEGORY_LABELS, get_settings  # noqa: E402

# ---- colour roles -----------------------------------------------------------
# Data colours: validated categorical palette (dark steps, checked against #121a2e with
# the dataviz validator: lightness band, chroma, CVD dE>=8.4 adjacent, contrast >=3:1).
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
POSITIVE, NEGATIVE = "#3987e5", "#e66767"   # diverging pair (blue <-> red)
NEUTRAL = "#4b5563"
# colour follows the entity: each category has a FIXED colour (never assigned by rank).
# Ten categories share eight hues, so 'support' reuses blue with a dotted line as a secondary
# encoding and 'other' is neutral grey.
CATEGORY_COLOR = {"bugs": SERIES[0], "performance": SERIES[1], "servers": SERIES[2], "balance": SERIES[3],
                  "monetization": SERIES[4], "matchmaking": SERIES[5], "ux": SERIES[6], "content": SERIES[7],
                  "support": SERIES[0], "other": "#8a93a6"}
CATEGORY_DASH = {"support": "dot"}
# UI accents (chrome only, never data): cyan / violet / teal
ACCENT, ACCENT2, ACCENT3 = "#22d3ee", "#a78bfa", "#2dd4bf"
SURFACE, SURFACE2, BORDER = "#0b1020", "#121a2e", "#22304d"
TEXT, TEXT2, MUTED = "#e6ebf5", "#aab4c8", "#7c879c"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', system-ui, -apple-system, Segoe UI, sans-serif; }}
.block-container {{ padding-top: 5rem; padding-bottom: 2rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{ background: #0e1528; border-right: 1px solid {BORDER}; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.01em; }}
.gp-title {{ font-size: 1.65rem; font-weight: 700; color: {TEXT}; margin-bottom: 0.1rem; }}
.gp-sub {{ color: {TEXT2}; font-size: 0.92rem; margin-bottom: 1.1rem; }}
.gp-badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 600;
            border: 1px solid {BORDER}; color: {TEXT2}; background: {SURFACE2}; margin-right: 6px; }}
.gp-badge.warn {{ color: #fab219; border-color: #5a4512; }}
.gp-badge.ok {{ color: {ACCENT3}; border-color: #12463f; }}
.gp-kpi {{ background: linear-gradient(180deg, {SURFACE2} 0%, #0f172b 100%); border: 1px solid {BORDER};
          border-radius: 14px; padding: 16px 18px; height: 100%; }}
.gp-kpi .lab {{ color: {MUTED}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.gp-kpi .val {{ color: {TEXT}; font-size: 1.75rem; font-weight: 700; line-height: 1.25; margin-top: 4px; }}
.gp-kpi .sub {{ color: {TEXT2}; font-size: 0.8rem; margin-top: 2px; }}
.gp-kpi .bar {{ height: 3px; border-radius: 3px; margin-bottom: 10px; width: 36px; }}
.gp-card {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 18px; margin-bottom: 12px; }}
.gp-insight {{ border-left: 3px solid {ACCENT}; padding: 8px 14px; background: #0f1a30; border-radius: 0 10px 10px 0;
              color: {TEXT}; margin-bottom: 8px; font-size: 0.92rem; }}
.gp-review {{ background: #0f172b; border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }}
.gp-review .meta {{ color: {MUTED}; font-size: 0.75rem; margin-bottom: 4px; }}
.gp-review .txt {{ color: {TEXT}; font-size: 0.9rem; }}
.gp-foot {{ color: {MUTED}; font-size: 0.76rem; margin-top: 18px; }}
div[data-testid="stChatMessage"] {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 14px; }}
</style>
"""

pio.templates["gameops"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT2, size=12),
    title=dict(font=dict(color=TEXT, size=15), x=0, xanchor="left"),
    colorway=SERIES, hovermode="closest",
    hoverlabel=dict(bgcolor="#0f172b", bordercolor=BORDER, font=dict(color=TEXT, size=12)),
    xaxis=dict(gridcolor="#1b2640", linecolor=BORDER, zeroline=False, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor="#1b2640", linecolor=BORDER, zeroline=False, tickfont=dict(color=MUTED)),
    legend=dict(orientation="h", y=1.08, x=0, font=dict(color=TEXT2), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=8, r=8, t=48, b=8), bargap=0.25))
pio.templates.default = "gameops"


def setup(title: str, subtitle: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="gp-title">{title}</div><div class="gp-sub">{subtitle}</div>', unsafe_allow_html=True)


# ---- data access (cached, read-only) ------------------------------------------
@st.cache_resource(show_spinner="Loading analytics database...")
def store():
    from src.agents.tools import store as _s
    return _s()


@st.cache_resource(show_spinner="Loading vector indexes...")
def retriever():
    from src.agents.tools import retriever as _r
    return _r()


@st.cache_resource(show_spinner="Loading NLP models...")
def nlp_models():
    import joblib
    return joblib.load(get_settings().artifacts_dir / "nlp" / "nlp_models.joblib")


def read_json(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner=False)
def _auto_build() -> str:
    """Build the demo pipeline and evaluations once per server process.

    Generated artifacts are git-ignored, so a fresh clone (e.g. a Streamlit Community Cloud deployment) starts
    without them. The build runs in a SEPARATE Python process: it peaks at ~1.2 GB, and that memory is returned to
    the system when it exits, so the long-running dashboard process stays small (free hosting has tight limits).
    """
    import subprocess
    for script in ("build_pipeline.py", "run_evaluations.py"):
        # fixed script names and the current interpreter only - no user input reaches this call
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, capture_output=True, text=True)  # noqa: S603
        if res.returncode != 0:
            raise RuntimeError(f"{script} failed: {res.stderr[-800:]}")
    return "built"


def ensure_built() -> bool:
    if (get_settings().artifacts_dir / "gameops.duckdb").exists() and (get_settings().artifacts_dir / "eval" / "agent_eval.json").exists():
        return True
    with st.status("First start: generating the synthetic demo data, training models, building the search index and "
                   "running evaluations (about 1-2 minutes, once per server)...", expanded=False) as status:
        try:
            _auto_build()
        except Exception as exc:  # show a clear message instead of a stack trace
            status.update(label="Automatic build failed", state="error")
            st.error(f"Build failed: {type(exc).__name__}: {exc}. Run `python scripts/build_pipeline.py` locally.")
            return False
        status.update(label="Build complete", state="complete")
    store.clear()
    retriever.clear()
    nlp_models.clear()
    return True


def filters(show_sentiment: bool = False) -> dict:
    """Global sidebar filters (persisted in session state across pages)."""
    s = store()
    games = {"All games": None, **{f"{n} ({g})": g for g, n in s.games}}
    st.sidebar.markdown("#### Filters")
    label = st.sidebar.selectbox("Game", list(games), key="f_game")
    default = (s.min_date, s.max_date)
    rng = st.sidebar.date_input("Date range", value=st.session_state.get("f_dates", default), min_value=s.min_date,
                                max_value=s.max_date, key="f_dates")
    start, end = (rng if isinstance(rng, (list, tuple)) and len(rng) == 2 else default)
    out = {"game_id": games[label], "game_label": label, "start": start, "end": end}
    if show_sentiment:
        out["sentiment"] = st.sidebar.radio("Recommendation", ["all", "negative", "positive"], horizontal=True, key="f_sent")
    st.sidebar.markdown(f'<div class="gp-foot">Data window {s.min_date} to {s.max_date}. Relative dates anchor to the '
                        f'latest review.</div>', unsafe_allow_html=True)
    return out


def data_source_badge() -> None:
    m = read_json("artifacts/pipeline_manifest.json") or {}
    if m.get("dataset") == "demo":
        st.markdown('<span class="gp-badge warn">SYNTHETIC DEMO DATA - fictional games, not real players</span>'
                    f'<span class="gp-badge">embedder: {m.get("embedder", "?")}</span>'
                    f'<span class="gp-badge">built {str(m.get("built_at", "?"))[:16].replace("T", " ")} UTC</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="gp-badge ok">dataset: {m.get("dataset")}</span>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "", color: str = ACCENT) -> None:
    col.markdown(f'<div class="gp-kpi"><div class="bar" style="background:{color}"></div><div class="lab">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def review_card(r: dict) -> None:
    import html
    rec = "recommended" if r.get("recommended", r.get("voted_up")) else "not recommended"
    color = POSITIVE if rec == "recommended" else NEGATIVE
    st.markdown(f'<div class="gp-review"><div class="meta"><b style="color:{color}">●</b> {rec} · {r.get("game", r.get("game_name", ""))}'
                f' · {str(r.get("date", r.get("created_at", "")))[:10]} · <code>{r.get("id", r.get("review_id"))}</code></div>'
                f'<div class="txt">{html.escape(str(r.get("text", r.get("review_text", ""))))}</div></div>', unsafe_allow_html=True)


def empty(msg: str = "No reviews match the current filters.") -> None:
    st.info(msg)


def cat_label(c: str) -> str:
    return CATEGORY_LABELS.get(c, c)


def iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


__all__ = ["CATEGORIES", "CATEGORY_COLOR", "CATEGORY_DASH", "SERIES", "POSITIVE", "NEGATIVE", "NEUTRAL", "ACCENT", "ACCENT2", "ACCENT3", "TEXT", "TEXT2",
           "MUTED", "BORDER", "setup", "store", "retriever", "nlp_models", "read_json", "filters", "kpi", "review_card",
           "empty", "cat_label", "iso", "data_source_badge", "ensure_built"]
