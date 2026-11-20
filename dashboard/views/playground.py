import pandas as pd
import streamlit as st
from common import cat_label, ensure_built, nlp_models, retriever, review_card, setup

from src.config import CATEGORIES
from src.nlp.rules import label as rule_label

setup("Model Playground", "Run the implemented models on your own text and inspect similar reviews from the index.")
if not ensure_built():
    st.stop()
nlp = nlp_models()
text = st.text_area("Review text", "Since the last patch the servers disconnect me every match and the new store feels pay to win.",
                    height=110, max_chars=2000)
if not text.strip():
    st.stop()
specific = [c for c in CATEGORIES if c != "other"]
rows = []
for name, model in nlp["all_models"].items():
    pred = model.predict([text])[0]
    rows.append({"model": name + (" (selected)" if name == nlp["selected"] else ""),
                 "predicted categories": ", ".join(cat_label(c) for c, v in zip(specific, pred, strict=False) if v) or "none"})
st.markdown("#### Complaint categories - model comparison")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
hits = [k for k, v in rule_label(text).items() if v]
st.caption("Keyword rules matched: " + (", ".join(cat_label(h) for h in hits) or "nothing") +
           ". Models were trained/evaluated on SYNTHETIC demo labels - treat outputs as illustrative.")

p_neg = float(nlp["sentiment_model"].predict_proba([text])[0, 1])
thr = nlp["sentiment_threshold"]
st.markdown("#### Text sentiment model")
st.progress(min(max(p_neg, 0.0), 1.0), text=f"P(not recommended) = {p_neg:.3f} → "
            f"{'negative' if p_neg >= thr else 'positive'} (threshold {thr:.2f}, tuned on validation)")

st.markdown("#### Similar reviews (hybrid retrieval)")
k = st.slider("Top K", 1, 10, 5)
r = retriever().search_reviews(text, top_k=k)
st.caption(f"best relevance {r['best_score']:.3f} (no-answer threshold {r['min_score']:.3f}) · mode {r['mode']} · "
           f"{r['latency_ms']:.1f} ms · embedder {r['embedder']}")
if not r["answerable"]:
    st.info("No sufficiently similar reviews - the retriever declined to return evidence.")
else:
    res = r["results"]
    st.dataframe(res[["source_id", "game_name", "dense_score", "lexical_score", "score", "categories"]].round(3),
                 hide_index=True, width="stretch")
    for x in res.to_dict("records"):
        review_card({"id": x["source_id"], "game": x["game_name"], "date": x["created_at"], "recommended": x["voted_up"], "text": x["text"]})
