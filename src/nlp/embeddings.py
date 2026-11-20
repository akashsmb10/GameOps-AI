"""Text embedders behind one interface. All return L2-normalised float32 vectors, so the
inner product equals cosine similarity (used by FAISS IndexFlatIP).

Order tried by `get_embedder("auto")`:
1. Sentence-Transformers (default all-MiniLM-L6-v2, 384-d) - the intended production
   embedder. Requires the model weights (downloaded once from Hugging Face or a local path).
2. spaCy `en_core_web_md` static word vectors (300-d mean of GloVe-style vectors) - real
   pretrained vectors, installable from GitHub; weaker than a sentence transformer.
3. TF-IDF + truncated SVD (LSA) fitted on the corpus - no download at all; lexical only.
The active embedder's name is stored with every index so results are reproducible.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger("gameops.embeddings")


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, local_only: bool = True):
        from sentence_transformers import SentenceTransformer
        # local_only avoids slow retries when the hub is unreachable; set
        # GAMEOPS_ALLOW_MODEL_DOWNLOAD=1 (or pass a local path) to fetch weights.
        self.model = SentenceTransformer(model_name, device="cpu", local_files_only=local_only)
        self.name = f"sentence-transformers:{model_name}"
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.model.encode(list(texts), batch_size=64, show_progress_bar=False))


class SpacyVectorEmbedder:
    def __init__(self, model: str = "en_core_web_md"):
        import spacy
        self.nlp = spacy.load(model, exclude=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler", "senter"])
        self.name = f"spacy:{model}:{self.nlp.meta.get('version', '?')}"
        self.dim = int(self.nlp.vocab.vectors_length)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = [d.vector for d in self.nlp.pipe((t.lower() for t in texts), batch_size=256)]
        return _normalise(np.vstack(vecs) if vecs else np.zeros((0, self.dim)))


class TfidfSvdEmbedder:
    def __init__(self, dim: int = 256, seed: int = 42):
        self.dim, self.seed, self.name, self._fitted = dim, seed, f"tfidf-svd:{dim}", False

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=self.seed).fit(X)
        self.dim, self._fitted = self.svd.n_components, True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedder must be fitted on the corpus first")
        return _normalise(self.svd.transform(self.vec.transform(texts)))


def _st_available_locally(model: str) -> bool:
    """True if `model` is a local directory or already in the Hugging Face cache."""
    import os
    from pathlib import Path
    if Path(model).is_dir():
        return True
    if os.getenv("GAMEOPS_ALLOW_MODEL_DOWNLOAD") == "1":
        return True
    try:
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            return False
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model, "modules.json"), str)
    except Exception:
        return False


@lru_cache(maxsize=4)
def get_embedder(kind: str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if kind == "auto" and not _st_available_locally(st_model):
        log.info("Sentence-Transformers model %s not available locally; skipping", st_model)
    elif kind in ("auto", "sentence-transformers"):
        try:
            import os
            return SentenceTransformerEmbedder(st_model, local_only=os.getenv("GAMEOPS_ALLOW_MODEL_DOWNLOAD") != "1"
                                               and kind == "auto")
        except Exception as exc:  # missing torch, blocked download, bad path
            if kind == "sentence-transformers":
                raise
            log.warning("Sentence-Transformers unavailable (%s); falling back", type(exc).__name__)
    if kind in ("auto", "spacy"):
        try:
            return SpacyVectorEmbedder()
        except Exception as exc:
            if kind == "spacy":
                raise
            log.warning("spaCy vectors unavailable (%s); falling back to TF-IDF/SVD", type(exc).__name__)
    return TfidfSvdEmbedder()
