import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (CATEGORY_COLOR, CATEGORY_DASH, SERIES, cat_label, data_source_badge, empty, ensure_built, filters, review_card, setup, store)
from sklearn.feature_extraction.text import CountVectorizer

setup("Complaint Intelligence", "What negative reviews are about, how that changes over time, and the evidence behind it.")
if not ensure_built():
    st.stop()
f = filters()
s = store()
data_source_badge()
scope = st.segmented_control("Base", ["negative", "all"], default="negative", key="comp_scope",
                             format_func=lambda x: "Negative reviews" if x == "negative" else "All reviews") or "negative"

comp = s.complaints(f["game_id"], f["start"], f["end"], scope)
if comp["base_reviews"] == 0:
    empty()
    st.stop()
cats = [c for c in comp["categories"] if c["reviews"] > 0]
a, b = st.columns([1, 1.35])
with a:
    d = cats[::-1]
    err = dict(type="data", symmetric=False, color="#7c879c", thickness=1, width=3,
               array=[100 * (x["share_95ci"][1] - x["share"]) for x in d],
               arrayminus=[100 * (x["share"] - x["share_95ci"][0]) for x in d])
    fig = go.Figure(go.Bar(x=[100 * x["share"] for x in d], y=[x["label"] for x in d], orientation="h", error_x=err,
                           marker_color=SERIES[0], marker_cornerradius=4, customdata=[x["reviews"] for x in d],
                           hovertemplate="%{y}<br>%{x:.1f}% (%{customdata} reviews)<extra></extra>"))
    fig.update_layout(title=f"Category share of {comp['base_reviews']:,} {scope} reviews (95% CI)", height=430,
                      xaxis_title="% of reviews (multi-label)")
    st.plotly_chart(fig, width="stretch")
with b:
    freq = st.segmented_control("Trend granularity", ["week", "month"], default="week", key="comp_freq") or "week"
    default = [c["category"] for c in cats[:4]]
    chosen = st.multiselect("Categories to plot (max 4)", [c["category"] for c in cats], default=default,
                            format_func=cat_label, max_selections=4, key="comp_cats")
    tr = s.complaint_trends(f["game_id"], f["start"], f["end"], freq, scope)
    fig = go.Figure()
    for c in chosen:  # fixed colour per category, so filtering never repaints a line
        t = tr[tr["category"] == c]
        fig.add_scatter(x=t["period"], y=100 * t["share"], mode="lines+markers", name=cat_label(c),
                        line=dict(width=2, color=CATEGORY_COLOR[c], dash=CATEGORY_DASH.get(c, "solid")), marker=dict(size=8),
                        customdata=t[["count", "reviews"]], hovertemplate=f"{cat_label(c)}<br>%{{x|%d %b}}: %{{y:.1f}}%"
                        "<br>%{customdata[0]} of %{customdata[1]}<extra></extra>")
    fig.update_layout(title="Complaint frequency over time", yaxis_title="% of reviews", height=360, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

if f["game_id"] is None:
    rows = []
    for gid, name in s.games:
        cg = s.complaints(gid, f["start"], f["end"], scope)
        rows += [{"game": name, "category": x["label"], "share": 100 * (x["share"] or 0), "n": x["reviews"]} for x in cg["categories"]]
    m = pd.DataFrame(rows).pivot(index="category", columns="game", values="share")
    n = pd.DataFrame(rows).pivot(index="category", columns="game", values="n")
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, customdata=n.values, xgap=2, ygap=2,
                               colorscale=[[0, "#0f1a30"], [0.5, "#256abf"], [1, "#86b6ef"]],
                               colorbar=dict(title="%"), hovertemplate="%{y} · %{x}<br>%{z:.1f}% (%{customdata} reviews)<extra></extra>"))
    fig.update_layout(title=f"Category share by game ({scope} reviews)", height=420)
    st.plotly_chart(fig, width="stretch")

st.markdown("#### Drill into a category")
cat = st.selectbox("Category", [c["category"] for c in cats], format_func=cat_label, key="comp_drill")
reps = s.representative_reviews(cat, f["game_id"], f["start"], f["end"], limit=200, negative_only=(scope == "negative"))
left, right = st.columns([1, 1.4])
with left:
    st.markdown("**Common issue phrases** (most frequent 2-3 word phrases in this category)")
    if len(reps) >= 5:
        cv = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=2)
        try:
            X = cv.fit_transform(reps["review_text"].str.lower())
            freq_df = pd.DataFrame({"phrase": cv.get_feature_names_out(), "reviews": (X > 0).sum(axis=0).A1})
            st.dataframe(freq_df.sort_values("reviews", ascending=False).head(12), hide_index=True, width="stretch")
        except ValueError:
            st.caption("Not enough repeated phrases.")
    else:
        st.caption("Too few reviews for phrase statistics.")
with right:
    st.markdown("**Representative reviews** (deduplicated, most helpful first)")
    if reps.empty:
        empty()
    for r in reps.head(5).to_dict("records"):
        review_card(r)

emerging = s.emerging_issues(f["game_id"], end=f["end"])
st.markdown("#### Emerging topics (latest 28 days vs previous 28 days)")
if emerging:
    st.dataframe([{"Category": e["label"], "Before": f"{100 * e['share_a']:.1f}%", "Latest": f"{100 * e['share_b']:.1f}%",
                   "Change (pp)": e["change_pct_points"], "p-value": round(e["p_value"], 5)} for e in emerging],
                 hide_index=True, width="stretch")
else:
    st.caption("No category increased significantly (p < 0.05).")
st.markdown('<div class="gp-foot">Categories are predicted by the selected complaint model (see AI Evaluation page for its '
            'measured accuracy). Phrase counts are descriptive; word clouds are deliberately not used.</div>', unsafe_allow_html=True)
