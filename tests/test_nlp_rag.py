import json

import numpy as np
import pandas as pd
import pytest

from src.agents.tools import retriever
from src.analytics.store import InvalidQuery
from src.config import get_settings
from src.nlp.embeddings import TfidfSvdEmbedder, get_embedder
from src.nlp.rules import label
from src.rag.answer import answer, extractive_answer, validate_citations
from src.rag.index import chunk_text


def test_rules_handle_negation():
    assert label("No crashes so far, runs fine.")["bugs"] == 0
    assert label("The game crashes every hour.")["bugs"] == 1
    assert label("Support never answered my ticket.")["support"] == 1


def test_embeddings_are_normalised():
    e = get_embedder(get_settings().embedder)
    v = e.encode(["servers are down", "great fun"]) if not isinstance(e, TfidfSvdEmbedder) else None
    if v is not None:
        assert v.shape[0] == 2 and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
    t = TfidfSvdEmbedder(dim=8).fit(["a server crash", "server lag", "fun game", "crash bug", "lag spike online"])
    assert np.allclose(np.linalg.norm(t.encode(["server crash"]), axis=1), 1, atol=1e-4)


def test_nlp_metrics_are_computed_and_selection_uses_validation():
    m = json.loads((get_settings().artifacts_dir / "nlp" / "nlp_metrics.json").read_text())
    sel = m["selection"]
    assert sel["selected"] == max(sel["validation"], key=lambda k: sel["validation"][k]["micro_f1"])
    assert 0 <= m["complaints_test"][sel["selected"]]["micro_f1"] <= 1
    assert m["sentiment_test"]["confusion_matrix"]["labels"] == ["positive", "negative"]


def test_chunking_long_reviews():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text)
    assert len(chunks) > 1 and all(len(c.split()) <= 80 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]  # 20-word overlap


def test_retrieval_filters_game_date_and_sentiment():
    r = retriever()
    out = r.search_reviews("servers disconnecting lag", game_id="G001", start="2026-07-01", end="2026-07-31",
                           sentiment="negative", top_k=8)
    res = out["results"]
    assert len(res) > 0
    assert set(res["game_id"]) == {"G001"} and not res["voted_up"].any()
    d = pd.to_datetime(res["created_at"]).dt.date
    assert d.min() >= pd.Timestamp("2026-07-01").date() and d.max() <= pd.Timestamp("2026-07-31").date()
    assert res["source_id"].is_unique


def test_invalid_retrieval_inputs():
    with pytest.raises(InvalidQuery):
        retriever().search_reviews("lag", game_id="G777")
    with pytest.raises(InvalidQuery):
        retriever().search_reviews("   ")


def test_citations_are_validated_and_invalid_ones_removed():
    out = validate_citations("A [R:X1] and [R:FAKE] and [D:DOC1]", {"R:X1", "D:DOC1"})
    assert out["valid"] == ["R:X1", "D:DOC1"] and out["invalid"] == ["R:FAKE"] and "FAKE" not in out["text"]


def test_rag_answer_cites_only_retrieved_evidence():
    r = retriever()
    ret = r.search_reviews("crashes and bugs", top_k=4)
    a = answer("What do players say about crashes?", ret)
    ids = {f"R:{i}" for i in ret["results"]["source_id"]}
    assert a["citations"] and set(c for c in a["citations"] if c.startswith("R:")) <= ids
    assert a["invalid_citations"] == []


def test_no_answer_when_nothing_retrieved():
    empty = {"results": pd.DataFrame(columns=["source_id"]), "answerable": False, "best_score": 0.0}
    assert answer("anything", empty)["mode"] == "no_answer"
    assert "could not find" in extractive_answer("q", pd.DataFrame())


def test_doc_search_finds_patch_note_by_version():
    d = retriever().search_docs("What changed in Starfall Arena update 2.1?")
    assert d["results"]["source_id"].iloc[0] == "PATCH-G001-2.1"
