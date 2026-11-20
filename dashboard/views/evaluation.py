import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import ACCENT2, ACCENT3, SERIES, ensure_built, kpi, read_json, setup

from src.monitoring import telemetry

setup("AI Evaluation & Monitoring", "Measured results from the evaluation suites and local telemetry - nothing here is hard-coded.")
if not ensure_built():
    st.stop()
rag, agent, nlp = read_json("artifacts/eval/rag_eval.json"), read_json("artifacts/eval/agent_eval.json"), read_json("artifacts/nlp/nlp_metrics.json")
mon = telemetry.summary()

st.markdown("#### Retrieval (RAG)")
if rag:
    c = st.columns(5)
    kpi(c[0], "Precision@5", f"{rag['review_precision_at_k']:.3f}", f"{rag['review_questions']} review questions")
    kpi(c[1], "Doc hit@3", f"{rag['doc_hit_at_3']:.3f}", f"{rag['doc_questions']} doc questions", ACCENT2)
    kpi(c[2], "Citation correctness", f"{rag['citation_correctness']:.3f}", "cited ids in evidence", ACCENT3)
    kpi(c[3], "Groundedness", f"{rag['groundedness']:.3f}", f"unsupported {rag['unsupported_claim_rate']:.3f}")
    kpi(c[4], "Off-topic refused", f"{rag['unanswerable_refused']:.2f}", f"{rag['unanswerable_questions']} off-topic questions", "#e66767")
    ab = pd.DataFrame(rag["retrieval_ablation"]).T.reset_index().rename(columns={"index": "mode"})
    fig = go.Figure(go.Bar(x=ab["mode"], y=ab["p_at_k"], marker_color=[SERIES[0] if m == rag["selected_mode"] else "#2b3a5c" for m in ab["mode"]],
                           marker_cornerradius=4, text=ab["p_at_k"].round(3), textposition="outside",
                           hovertemplate="%{x}: precision@5 %{y:.3f}<extra></extra>"))
    fig.update_layout(title=f"Retrieval mode ablation (selected on dev set: {rag['selected_mode']}) - embedder {rag['embedder']}",
                      yaxis_title="precision@5", yaxis_range=[0, 1.1], height=300)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{rag['answer_mode']}. Relevance labels come from the synthetic generator.")
else:
    st.info("Run `python scripts/run_evaluations.py` to produce RAG metrics.")

st.markdown("#### Agent")
if agent:
    g = pd.DataFrame(agent["by_group"]).T.reset_index().rename(columns={"index": "group"})
    g["pass_rate"] = g["passed"] / g["count"]
    c = st.columns([1, 2.4])
    kpi(c[0], "Agent test cases passed", f"{agent['passed']}/{agent['cases']}", f"pass rate {agent['pass_rate']:.3f}")
    fig = go.Figure(go.Bar(y=g["group"], x=g["pass_rate"], orientation="h", marker_color=SERIES[2], marker_cornerradius=4,
                           customdata=g[["passed", "count"]], hovertemplate="%{y}: %{customdata[0]}/%{customdata[1]}<extra></extra>"))
    fig.update_layout(title="Pass rate by test group", xaxis_range=[0, 1.05], height=320)
    c[1].plotly_chart(fig, width="stretch")
    st.caption(agent.get("mode_note", ""))
    heldout = read_json("artifacts/eval/agent_heldout.json")
    if heldout:
        st.markdown(f'<div class="gp-insight">Held-out cases written after development: <b>{heldout["passed"]}/{heldout["cases"]}</b> passed '
                    '(reported without changing the code to fit them).</div>', unsafe_allow_html=True)

st.markdown("#### Complaint & sentiment models")
if nlp:
    rows = [{"model": k, "test micro-F1": v["micro_f1"], "test macro-F1": v["macro_f1"],
             "validation micro-F1": nlp["selection"]["validation"][k]["micro_f1"]} for k, v in nlp["complaints_test"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    pc = pd.DataFrame(nlp["complaints_test"][nlp["selection"]["selected"]]["per_category"]).round(3)
    st.dataframe(pc, hide_index=True, width="stretch")
    snt = nlp["sentiment_test"]
    st.caption(f"Selected: {nlp['selection']['selected']} (rule: {nlp['selection']['rule']}). Labels: {nlp['label_source']}. "
               f"Sentiment model ROC-AUC {snt['roc_auc']:.3f}, negative-class precision {snt['precision_negative']:.3f} / "
               f"recall {snt['recall_negative']:.3f}. {nlp['caveat']}")

st.markdown("#### Monitoring")
st.caption(mon["label"])
c = st.columns(4)
api = mon["api"]
kpi(c[0], "API requests", f"{api['requests']:,}", f"last {mon['window_hours']:.0f} h")
kpi(c[1], "API p95 latency", f"{api['p95_ms'] or 0:.1f} ms", f"p50 {api['p50_ms'] or 0:.1f} ms", ACCENT2)
kpi(c[2], "Server error rate", f"{100 * (api['error_rate'] or 0):.2f}%", "HTTP 5xx", "#e66767")
ag = mon["agent"]
kpi(c[3], "Tool-call success", "n/a" if ag["tool_success_rate"] is None else f"{100 * ag['tool_success_rate']:.1f}%",
    f"{ag['tool_calls']} calls, {ag['failed_tool_calls']} failed", ACCENT3)
if api["endpoints"]:
    ep = pd.DataFrame(api["endpoints"]).T.reset_index().rename(columns={"index": "endpoint"})
    st.dataframe(ep, hide_index=True, width="stretch")
m, d = mon["models"], mon["data"]
st.markdown(f'<div class="gp-card">Dataset <b>{d["dataset"]}</b> refreshed {d["dataset_refreshed_at"]} · '
            f'{d["included_reviews"]:,} included reviews<br>Complaint model <b>{m["complaint_model"]}</b> · embedder '
            f'<b>{m["embedder"]}</b> · retrieval mode <b>{m["retrieval_mode"]}</b><br>Index built {m["index_built_at"]} '
            f'({m["index_age_hours"]} h ago) · retrieval p95 {mon["retrieval"]["p95_ms"]} ms · citation validity '
            f'{ag["citation_validity"]}</div>', unsafe_allow_html=True)
bench = read_json("artifacts/eval/load_test.json")
if bench:
    st.markdown("#### Load test")
    st.dataframe(pd.DataFrame(bench["endpoints"]), hide_index=True, width="stretch")
    st.caption(bench["environment"])
