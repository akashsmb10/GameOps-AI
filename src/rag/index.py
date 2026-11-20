"""Chunking and FAISS index construction / search.

Two indexes:
* reviews - one chunk per review; reviews longer than 120 words are split into 80-word
  windows with 20-word overlap (each chunk keeps the parent review_id).
* docs    - update notes (one chunk each) and project knowledge docs (one chunk per
  '## ' section).
Vectors are L2-normalised, so IndexFlatIP scores are cosine similarities. Metadata
filtering uses FAISS IDSelectorBatch (exact search restricted to allowed ids).

Hybrid retrieval: dense (FAISS) and lexical (TF-IDF, English stop words removed) candidate
lists are merged with reciprocal-rank fusion (RRF, k=60). `relevance` = dense cosine +
lexical cosine of each hit; the best relevance drives the no-answer decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd

from src.config import CATEGORIES
from src.nlp.embeddings import TfidfSvdEmbedder

MAX_WORDS, WINDOW, OVERLAP = 120, 80, 20


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    step = WINDOW - OVERLAP
    return [" ".join(words[i:i + WINDOW]) for i in range(0, max(1, len(words) - OVERLAP), step)]


def review_chunks(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in reviews.itertuples(index=False):
        cats = [c for c in CATEGORIES if getattr(r, c, False)]
        for j, ch in enumerate(chunk_text(r.review_text)):
            rows.append({"chunk_id": f"{r.review_id}#{j}", "source_type": "review", "source_id": r.review_id,
                         "game_id": r.game_id, "game_name": r.game_name, "created_at": pd.Timestamp(r.created_at),
                         "voted_up": bool(r.voted_up), "categories": ",".join(cats), "text": ch,
                         "text_hash": hashlib.sha1(f"{r.game_id}|{ch.lower().strip()}".encode(), usedforsecurity=False).hexdigest()})
    return pd.DataFrame(rows)


def doc_chunks(notes: list[dict], knowledge_dir: Path) -> pd.DataFrame:
    rows = []
    for n in notes:
        text = f"{n['game_name']} update {n['version']} ({n['release_date']}): " + "; ".join(n["changes"])
        rows.append({"chunk_id": n["doc_id"], "source_type": "update_note", "source_id": n["doc_id"], "game_id": n["game_id"],
                     "game_name": n["game_name"], "created_at": pd.Timestamp(n["release_date"]), "text": text,
                     "title": f"{n['game_name']} {n['version']}"})
    for f in sorted(knowledge_dir.glob("*.md")):
        for sec in re.split(r"\n(?=## )", f.read_text()):
            title = sec.strip().splitlines()[0].lstrip("# ").strip()
            body = " ".join(sec.strip().splitlines()[1:]).strip()
            if body:
                sid = f"DOC-{f.stem}-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}"
                rows.append({"chunk_id": sid, "source_type": "project_doc", "source_id": sid, "game_id": None,
                             "game_name": None, "created_at": pd.NaT, "text": f"{title}. {body}", "title": title})
    return pd.DataFrame(rows)


class VectorIndex:
    def __init__(self, index: faiss.Index, meta: pd.DataFrame, embedder, manifest: dict, lexical=None):
        self.index, self.meta, self.embedder, self.manifest = index, meta.reset_index(drop=True), embedder, manifest
        if lexical is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # token pattern keeps version numbers such as "2.1" as single tokens
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english",
                                  token_pattern=r"(?u)\b\w+(?:\.\w+)*\b")  # noqa: S106 (regex, not a password)
            lexical = (vec, vec.fit_transform(self.meta["text"]))
        self.lex_vec, self.lex_X = lexical

    @classmethod
    def build(cls, meta: pd.DataFrame, embedder, name: str) -> "VectorIndex":
        if isinstance(embedder, TfidfSvdEmbedder) and not embedder._fitted:
            embedder.fit(meta["text"].tolist())
        vecs = embedder.encode(meta["text"].tolist())
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        manifest = {"name": name, "embedder": embedder.name, "dim": int(vecs.shape[1]), "vectors": int(index.ntotal),
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "faiss_index": "IndexFlatIP (exact cosine)"}
        return cls(index, meta, embedder, manifest)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self.manifest["name"]
        faiss.write_index(self.index, str(out_dir / f"{name}.faiss"))
        self.meta.to_parquet(out_dir / f"{name}_meta.parquet", index=False)
        (out_dir / f"{name}_manifest.json").write_text(json.dumps(self.manifest, indent=2))
        if isinstance(self.embedder, TfidfSvdEmbedder):
            joblib.dump(self.embedder, out_dir / f"{name}_embedder.joblib")
        joblib.dump((self.lex_vec, self.lex_X), out_dir / f"{name}_lexical.joblib")

    @classmethod
    def load(cls, out_dir: Path, name: str, embedder) -> "VectorIndex":
        manifest = json.loads((out_dir / f"{name}_manifest.json").read_text())
        emb_file = out_dir / f"{name}_embedder.joblib"
        if emb_file.exists():
            embedder = joblib.load(emb_file)
        if embedder.name != manifest["embedder"]:
            raise RuntimeError(f"Index {name} was built with {manifest['embedder']} but {embedder.name} is active; rebuild it")
        lex_file = out_dir / f"{name}_lexical.joblib"
        lexical = joblib.load(lex_file) if lex_file.exists() else None
        return cls(faiss.read_index(str(out_dir / f"{name}.faiss")), pd.read_parquet(out_dir / f"{name}_meta.parquet"),
                   embedder, manifest, lexical)

    FRAMING = {"players", "player", "saying", "say", "says", "think", "thinks", "feel", "reviews", "review", "people",
               "talk", "talking", "mention", "mentions", "tell", "telling", "game", "games"}

    def coverage(self, query: str, min_df: int = 2, ignore: set[str] | None = None) -> float:
        """Share of the query's content words (stop words removed) that occur in at least
        `min_df` indexed chunks. Used as the no-answer gate: off-topic questions mostly use
        words the corpus never contains."""
        analyzer = self.lex_vec.build_analyzer()
        skip = self.FRAMING | (ignore or set())  # game names are matched by metadata filters, not text
        words = [t for t in analyzer(query) if " " not in t and t not in skip]
        if not words:
            return 0.0
        if not hasattr(self, "_df"):
            X = self.lex_X.tocsc()
            self._df = dict(zip(self.lex_vec.get_feature_names_out(), (X > 0).sum(axis=0).A1, strict=False))
        return sum(self._df.get(w, 0) >= min_df for w in words) / len(words)

    def search(self, query: str, top_k: int = 5, allowed: np.ndarray | None = None, dedupe: bool = True,
               mode: str = "hybrid") -> pd.DataFrame:
        """mode: 'hybrid' (RRF of dense + lexical ranks), 'sum' (dense + lexical scores), 'dense' or 'lexical'."""
        n = self.index.ntotal
        fetch = min(n, max(top_k * 8, 40))
        q = self.embedder.encode([query])
        if allowed is not None and len(allowed) == 0:
            return self.meta.iloc[0:0].assign(score=[], dense_score=[], lexical_score=[])
        # dense candidates
        if allowed is not None:
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.asarray(allowed, dtype=np.int64)))
            d_scores, d_ids = self.index.search(q, min(fetch, len(allowed)), params=params)
        else:
            d_scores, d_ids = self.index.search(q, fetch)
        d_ids = d_ids[0][d_ids[0] >= 0]
        # lexical scores for every row (sparse product is fast at this scale)
        lex = np.asarray((self.lex_X @ self.lex_vec.transform([query]).T).todense()).ravel()
        if allowed is not None:
            mask = np.zeros(n, dtype=bool)
            mask[allowed] = True
            lex = np.where(mask, lex, -1.0)
        l_ids = np.argsort(-lex)[:fetch]
        l_ids = l_ids[lex[l_ids] > 0]
        dense_all = {int(i): float(s_) for i, s_ in zip(d_ids, d_scores[0][: len(d_ids)], strict=False)}
        if mode == "dense":
            order = list(d_ids)
        elif mode == "lexical":
            order = list(l_ids)
        elif mode == "sum":
            order = list(dict.fromkeys([int(i) for i in d_ids] + [int(i) for i in l_ids]))
        else:
            rrf: dict[int, float] = {}
            for rank, i in enumerate(d_ids):
                rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + rank)
            for rank, i in enumerate(l_ids):
                rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + rank)
            order = sorted(rrf, key=rrf.get, reverse=True)
        if not order:
            return self.meta.iloc[0:0].assign(score=[], dense_score=[], lexical_score=[])
        order = np.asarray(order, dtype=np.int64)
        missing = [i for i in order if int(i) not in dense_all]
        if missing:  # exact dense score for lexical-only candidates
            vecs = np.vstack([self.index.reconstruct(int(i)) for i in missing])
            for i, v in zip(missing, vecs @ q[0], strict=False):
                dense_all[int(i)] = float(v)
        res = self.meta.iloc[order].copy()
        res["dense_score"] = [dense_all[int(i)] for i in order]
        res["lexical_score"] = np.clip(lex[order], 0, None)
        res["score"] = res["dense_score"] + res["lexical_score"]
        if mode == "sum":
            res = res.sort_values("score", ascending=False, kind="stable")
        if dedupe and "text_hash" in res:
            res = res.drop_duplicates("text_hash", keep="first")
        if dedupe:
            res = res.drop_duplicates("source_id", keep="first")
        return res.head(top_k).reset_index(drop=True)
