import plotly.graph_objects as go
import streamlit as st
from common import (NEGATIVE, POSITIVE, SERIES, data_source_badge, empty, ensure_built, filters, kpi, review_card, setup,
                    store)

setup("Player Sentiment", "Source recommendations over time and by game. Model-predicted sentiment is shown separately.")
if not ensure_built():
    st.stop()
f = filters(show_sentiment=True)
s = store()
data_source_badge()
freq = st.segmented_control("Granularity", ["day", "week", "month"], default="week", key="sent_freq") or "week"

ov = s.overview(f["game_id"], f["start"], f["end"])
if ov["total_reviews"] == 0:
    empty()
    st.stop()
c = st.columns(3)
kpi(c[0], "Reviews", f"{ov['total_reviews']:,}")
kpi(c[1], "Recommended", f"{ov['positive_pct']:.1f}%", color=POSITIVE)
kpi(c[2], "Not recommended", f"{ov['negative_pct']:.1f}%", color=NEGATIVE)

ts = s.sentiment_timeseries(f["game_id"], f["start"], f["end"], freq)
a, b = st.columns(2)
with a:
    fig = go.Figure()
    fig.add_scatter(x=list(ts["period"]) + list(ts["period"][::-1]), y=list(100 * ts["ci_high"]) + list(100 * ts["ci_low"][::-1]),
                    fill="toself", fillcolor="rgba(230,103,103,0.15)", line=dict(width=0), hoverinfo="skip", name="95% CI")
    fig.add_scatter(x=ts["period"], y=100 * ts["negative_share"], mode="lines+markers", name="Negative share",
                    line=dict(color=NEGATIVE, width=2), marker=dict(size=8),
                    customdata=ts[["negative", "reviews"]], hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}% negative"
                    "<br>%{customdata[0]} of %{customdata[1]} reviews<extra></extra>")
    fig.update_layout(title="Negative share over time (with 95% CI)", yaxis_title="% not recommended", height=340,
                      hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, width="stretch")
with b:
    fig = go.Figure()
    fig.add_scatter(x=ts["period"], y=ts["positive"], name="Recommended", mode="lines", line=dict(color=POSITIVE, width=2))
    fig.add_scatter(x=ts["period"], y=ts["negative"], name="Not recommended", mode="lines", line=dict(color=NEGATIVE, width=2))
    fig.update_layout(title="Positive vs negative review counts", yaxis_title="reviews", height=340, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

bg = s.sentiment_by_game(f["start"], f["end"]).sort_values("negative_share")
fig = go.Figure()
fig.add_bar(y=bg["game_name"], x=100 * bg["negative_share"], orientation="h", name="Source label (not recommended)",
            marker_color=NEGATIVE, marker_cornerradius=4, customdata=bg["reviews"],
            hovertemplate="%{y}<br>%{x:.1f}% not recommended (%{customdata} reviews)<extra></extra>")
fig.add_bar(y=bg["game_name"], x=100 * bg["model_negative_share"], orientation="h", name="Model-predicted negative",
            marker_color=SERIES[6], marker_cornerradius=4,
            hovertemplate="%{y}<br>%{x:.1f}% predicted negative by the text model<extra></extra>")
fig.update_layout(title="Negative share by game: source label vs model prediction", barmode="group", height=320,
                  xaxis_title="% of reviews", bargap=0.35, bargroupgap=0.08)
st.plotly_chart(fig, width="stretch")
st.caption("The source label is the player's own recommendation. The model column is a TF-IDF classifier's prediction from "
           "text; differences show model error, not player opinion.")

st.markdown("#### Example reviews")
frame = s.reviews_frame(f["game_id"], f["start"], f["end"])
if f.get("sentiment") == "negative":
    frame = frame[~frame["voted_up"]]
elif f.get("sentiment") == "positive":
    frame = frame[frame["voted_up"]]
if frame.empty:
    empty()
else:
    for r in frame.sort_values(["helpful_votes", "created_at"], ascending=False).head(6).to_dict("records"):
        review_card(r)
    st.download_button("Download these reviews (CSV)", frame.to_csv(index=False), "gameops_sentiment_reviews.csv", "text/csv")
