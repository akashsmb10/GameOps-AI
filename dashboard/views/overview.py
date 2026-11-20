import plotly.graph_objects as go
import streamlit as st
from common import (ACCENT, ACCENT2, ACCENT3, NEGATIVE, POSITIVE, SERIES, data_source_badge, empty, ensure_built,
                    filters, kpi, read_json, setup, store)

setup("Executive Overview", "Review volume, sentiment and the complaints that matter most - for the selected game and dates.")
if not ensure_built():
    st.stop()
f = filters()
s = store()
data_source_badge()

ov = s.overview(f["game_id"], f["start"], f["end"])
if ov["total_reviews"] == 0:
    empty()
    st.stop()
dq = read_json("data/processed/data_quality_report.json") or {}
c = st.columns(5)
kpi(c[0], "Total reviews", f"{ov['total_reviews']:,}", f"{ov['start']} → {ov['end']}", ACCENT)
kpi(c[1], "Positive", f"{ov['positive_pct']:.1f}%", f"{ov['positive_reviews']:,} recommended", POSITIVE)
lo, hi = ov["negative_share_95ci"]
kpi(c[2], "Negative", f"{ov['negative_pct']:.1f}%", f"95% CI {100 * lo:.1f}–{100 * hi:.1f}%", NEGATIVE)
kpi(c[3], "Games analyzed", f"{ov['games_analyzed']}", "in this selection", ACCENT2)
kpi(c[4], "Latest data update", str(dq.get("generated_at", "?"))[:10], f"last review {str(ov['last_review'])[:10]}", ACCENT3)
if ov["small_sample"]:
    st.warning("Fewer than 30 reviews in this window - percentages are unreliable.")

st.markdown("&nbsp;")
left, right = st.columns([1.55, 1])
ts = s.sentiment_timeseries(f["game_id"], f["start"], f["end"], "week")
with left:
    fig = go.Figure()
    gap = dict(color="#0b1020", width=2)  # 2px surface gap between stacked segments
    fig.add_bar(x=ts["period"], y=ts["positive"], name="Recommended", marker_color=POSITIVE, marker_cornerradius=4, marker_line=gap,
                hovertemplate="Week of %{x|%d %b}<br>Recommended: %{y}<extra></extra>")
    fig.add_bar(x=ts["period"], y=ts["negative"], name="Not recommended", marker_color=NEGATIVE, marker_cornerradius=4, marker_line=gap,
                hovertemplate="Week of %{x|%d %b}<br>Not recommended: %{y}<extra></extra>")
    fig.update_layout(title="Weekly review volume (first/last weeks may be partial)", barmode="stack", height=340, yaxis_title="reviews",
                      bargap=0.3, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
with right:
    comp = s.complaints(f["game_id"], f["start"], f["end"], "negative")
    cats = [x for x in comp["categories"] if x["reviews"] > 0][:6][::-1]
    fig = go.Figure(go.Bar(x=[100 * x["share"] for x in cats], y=[x["label"] for x in cats], orientation="h",
                           marker_color=SERIES[0], marker_cornerradius=4,
                           customdata=[[x["reviews"], 100 * x["share_95ci"][0], 100 * x["share_95ci"][1]] for x in cats],
                           hovertemplate="%{y}<br>%{x:.1f}% of negative reviews (%{customdata[0]})"
                                         "<br>95% CI %{customdata[1]:.1f}–%{customdata[2]:.1f}%<extra></extra>",
                           text=[f"{100 * x['share']:.0f}%" for x in cats], textposition="outside", cliponaxis=False))
    fig.update_layout(title=f"Top complaints (of {comp['base_reviews']:,} negative reviews)", height=340,
                      xaxis_title="% of negative reviews", margin=dict(l=8, r=40, t=48, b=8))
    st.plotly_chart(fig, width="stretch")

st.markdown("#### Insights")
emerging = s.emerging_issues(f["game_id"], end=f["end"])
by_game = s.sentiment_by_game(f["start"], f["end"])
ins = []
if len(by_game) > 1 and f["game_id"] is None:
    w = by_game.sort_values("negative_share", ascending=False).iloc[0]
    ins.append(f"<b>{w.game_name}</b> has the highest negative share in this window: {100 * w.negative_share:.1f}% of {w.reviews:,} reviews.")
top = comp["categories"][0]
ins.append(f"The most common complaint is <b>{top['label']}</b>, in {100 * top['share']:.1f}% of negative reviews.")
if emerging:
    e = emerging[0]
    ins.append(f"Emerging: <b>{e['label']}</b> rose from {100 * e['share_a']:.1f}% to {100 * e['share_b']:.1f}% of reviews "
               f"({e['period_a']} vs {e['period_b']}, p={e['p_value']:.4f}). Check the Complaint Intelligence page for evidence.")
else:
    ins.append("No complaint category increased significantly in the latest 28 days versus the 28 days before.")
for i in ins:
    st.markdown(f'<div class="gp-insight">{i}</div>', unsafe_allow_html=True)

if emerging:
    st.markdown("#### Recently emerging issues")
    st.dataframe([{"Category": e["label"], "Previous 28 days": f"{100 * e['share_a']:.1f}%", "Latest 28 days": f"{100 * e['share_b']:.1f}%",
                   "Change (pp)": e["change_pct_points"], "p-value": round(e["p_value"], 5)} for e in emerging],
                 hide_index=True, width="stretch")

frame = s.reviews_frame(f["game_id"], f["start"], f["end"])
st.download_button("Download filtered reviews (CSV)", frame.to_csv(index=False), "gameops_reviews_filtered.csv", "text/csv")
st.markdown('<div class="gp-foot">Percentages describe the collected reviews, not all players. Sentiment = player recommendation '
            '(source label). Complaint categories are model predictions.</div>', unsafe_allow_html=True)
