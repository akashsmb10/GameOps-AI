from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (NEGATIVE, SERIES, data_source_badge, empty, ensure_built, kpi, read_json, review_card, setup, store)

setup("Game Update Impact", "Review trends around recorded update dates. Timing is an association - never proof of cause.")
if not ensure_built():
    st.stop()
s = store()
data_source_badge()
notes = s.update_notes()
if notes.empty:
    st.info("No update metadata is available for this dataset, so update-specific analysis is not shown.")
    st.stop()
manifest = read_json("artifacts/pipeline_manifest.json") or {}
if manifest.get("dataset") == "demo":
    st.caption("Update notes here are fictional patch notes from the synthetic demo generator.")

notes["label"] = notes["game_name"] + " · v" + notes["version"].astype(str) + " · " + notes["release_date"].dt.date.astype(str)
choice = st.selectbox("Update", notes["label"].tolist(), key="upd_choice")
n = notes[notes["label"] == choice].iloc[0]
days = st.slider("Window before / after (days)", 7, 35, 21, key="upd_days")
rel = n["release_date"].date()
a0, a1, b0, b1 = rel - timedelta(days=days), rel - timedelta(days=1), rel, rel + timedelta(days=days - 1)
if a0 < s.min_date or b1 > s.max_date:
    st.warning("Part of the window falls outside the data range; the comparison uses the available days only.")
st.markdown(f'<div class="gp-card"><b>Update notes</b> <code>{n["doc_id"]}</code><br>{n["changes"]}</div>', unsafe_allow_html=True)

cmp = s.compare_periods(n["game_id"], a0, a1, b0, b1)
A, B = cmp["period_a"], cmp["period_b"]
if min(A["reviews"], B["reviews"]) == 0:
    empty("One of the windows has no reviews.")
    st.stop()
c = st.columns(4)
kpi(c[0], "Reviews before", f"{A['reviews']:,}", f"{A['start']} → {A['end']}")
kpi(c[1], "Reviews after", f"{B['reviews']:,}", f"{B['start']} → {B['end']}")
kpi(c[2], "Negative before", f"{A['negative_pct']:.1f}%", f"{A['negative']:,} of {A['reviews']:,} reviews", SERIES[0])
p = cmp["negative_share_p_value"]
kpi(c[3], "Negative after", f"{B['negative_pct']:.1f}%", f"{cmp['negative_share_change_pct_points']:+.1f} pp · p={p:.4f}", NEGATIVE)

ts = s.sentiment_timeseries(n["game_id"], rel - timedelta(days=days * 2), min(rel + timedelta(days=days * 2), s.max_date), "day")
ts["roll"] = ts["negative"].rolling(7, min_periods=3).sum() / ts["reviews"].rolling(7, min_periods=3).sum()
fig = go.Figure()
fig.add_bar(x=ts["period"], y=ts["reviews"], name="Daily reviews", marker_color="#2b3a5c", marker_cornerradius=3,
            hovertemplate="%{x|%d %b}: %{y} reviews<extra></extra>")
fig.update_layout(title=f"Daily review volume around v{n['version']}", height=260, showlegend=False, yaxis_title="reviews")
fig.add_vline(x=pd.Timestamp(rel), line=dict(color="#a78bfa", dash="dash", width=1.5))
st.plotly_chart(fig, width="stretch")
fig = go.Figure()
fig.add_scatter(x=ts["period"], y=100 * ts["roll"], mode="lines", line=dict(color=NEGATIVE, width=2), name="7-day negative share",
                hovertemplate="%{x|%d %b}: %{y:.1f}% negative (7-day)<extra></extra>")
fig.add_vline(x=pd.Timestamp(rel), line=dict(color="#a78bfa", dash="dash", width=1.5))
fig.add_annotation(x=pd.Timestamp(rel), y=1, yref="paper", text=f"v{n['version']} released", showarrow=False,
                   font=dict(color="#a78bfa"), xanchor="left", yanchor="bottom")
fig.update_layout(title="Negative share, 7-day rolling", height=300, yaxis_title="% not recommended", showlegend=False)
st.plotly_chart(fig, width="stretch")

ch = pd.DataFrame(cmp["category_changes"])
ch = ch[(ch["share_a"] > 0) | (ch["share_b"] > 0)].sort_values("change_pct_points")
fig = go.Figure()
fig.add_bar(y=ch["label"], x=100 * ch["share_a"], orientation="h", name="Before", marker_color=SERIES[0], marker_cornerradius=4)
fig.add_bar(y=ch["label"], x=100 * ch["share_b"], orientation="h", name="After", marker_color=SERIES[1], marker_cornerradius=4,
            customdata=ch[["change_pct_points", "p_value"]], hovertemplate="%{y}<br>after: %{x:.1f}% (%{customdata[0]:+.1f} pp, "
            "p=%{customdata[1]:.4f})<extra></extra>")
fig.update_layout(title="Complaint share of all reviews: before vs after", barmode="group", height=420,
                  xaxis_title="% of reviews", bargroupgap=0.08)
st.plotly_chart(fig, width="stretch")
sig = ch[(ch["p_value"] < 0.05) & (ch["change_pct_points"] > 0)].sort_values("change_pct_points", ascending=False)
st.markdown("#### Supporting evidence after the update")
if sig.empty:
    st.caption("No category increased significantly; showing the most helpful negative reviews after the update.")
    cat = ch.sort_values("change_pct_points").iloc[-1]["category"]
else:
    cat = sig.iloc[0]["category"]
    st.markdown(f'<div class="gp-insight">Largest significant increase: <b>{sig.iloc[0]["label"]}</b> '
                f'({sig.iloc[0]["change_pct_points"]:+.1f} pp). This coincides with the release; the data cannot show the '
                f'update caused it (seasonality, events or review campaigns could also explain it).</div>', unsafe_allow_html=True)
reps = s.representative_reviews(cat, n["game_id"], b0, b1, limit=4)
for r in reps.to_dict("records"):
    review_card(r)
