"""Complaint-category (multi-label) and sentiment models, with honest evaluation.

Labels:
* demo dataset   -> generator labels (ground truth ONLY for the synthetic set). Test reviews
                    use held-out phrase templates, so scores reflect unseen wording.
* steam dataset  -> no ground truth exists; the TF-IDF model is trained on weak keyword
                    labels and its "scores" are agreement with those weak labels, not
                    accuracy. Add manually reviewed labels in data/labels/manual_labels.csv
                    for a real evaluation (see docs/DATA_DICTIONARY.md).
Sentiment: the SOURCE label is the player's recommendation (voted_up). The model sentiment
is a TF-IDF logistic regression trained to predict it from text; both are kept separately.
Model selection and thresholds use a validation split carved from training data; the test
split is evaluated once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.config import CATEGORIES
from src.nlp import rules

SPECIFIC = [c for c in CATEGORIES if c != "other"]
PROTOTYPES = {
    "bugs": ["the game crashes and has many bugs", "glitches, errors and a corrupted save file", "the game freezes"],
    "performance": ["low frame rate and stuttering", "poor optimization, the game runs slowly", "long loading times"],
    "servers": ["servers are down and I keep disconnecting", "high ping, lag and connection problems online"],
    "balance": ["weapons and characters are badly balanced", "unfair difficulty, overpowered and underpowered builds"],
    "monetization": ["too many microtransactions and pay to win", "expensive premium store and loot boxes"],
    "matchmaking": ["matchmaking takes too long", "matched against much higher ranked players"],
    "ux": ["the menus and user interface are confusing", "bad controls, HUD and accessibility options"],
    "content": ["not enough content, repetitive and nothing to do", "no new maps, missions or updates"],
    "support": ["customer support never answers", "refund request and ban appeal ignored by support"],
}


def _bucket(review_id: str) -> float:
    return int(hashlib.md5(review_id.encode(), usedforsecurity=False).hexdigest()[:8], 16) / 0xFFFFFFFF


def tune_thresholds(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 91)
    out = []
    for j in range(y.shape[1]):
        f1s = [f1_score(y[:, j], scores[:, j] >= t, zero_division=0) for t in grid]
        out.append(grid[int(np.argmax(f1s))])
    return np.array(out)


class TfidfComplaintModel:
    name = "tfidf_logreg"

    def fit(self, texts, y: np.ndarray, val_texts, y_val: np.ndarray):
        self.pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                              ("clf", OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0)))])
        self.pipe.fit(texts, y)
        self.thresholds = tune_thresholds(y_val, self.scores(val_texts))
        return self

    def scores(self, texts) -> np.ndarray:
        return self.pipe.predict_proba(list(texts))

    def predict(self, texts) -> np.ndarray:
        return (self.scores(texts) >= self.thresholds).astype(int)

    def top_terms(self, k: int = 8) -> dict[str, list[str]]:
        vocab = np.array(self.pipe.named_steps["tfidf"].get_feature_names_out())
        return {c: vocab[np.argsort(-est.coef_[0])[:k]].tolist()
                for c, est in zip(SPECIFIC, self.pipe.named_steps["clf"].estimators_, strict=False)}


class HybridComplaintModel(TfidfComplaintModel):
    """TF-IDF features plus the nine keyword-rule indicators in one logistic model: the rules
    contribute domain vocabulary that generalises, TF-IDF contributes context."""
    name = "tfidf_plus_rules"

    def _features(self, texts):
        from scipy.sparse import csr_matrix, hstack
        texts = list(texts)
        return hstack([self.vec.transform(texts), csr_matrix(RuleModel().predict(texts) * 3.0)]).tocsr()

    def fit(self, texts, y, val_texts, y_val):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(list(texts))
        self.clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
        self.clf.fit(self._features(texts), y)
        self.thresholds = tune_thresholds(y_val, self.scores(val_texts))
        return self

    def scores(self, texts):
        return self.clf.predict_proba(self._features(texts))

    def top_terms(self, k: int = 8):
        return {}


class EmbeddingZeroShotModel:
    """Cosine similarity to short category descriptions; only the per-category threshold is
    tuned on labelled validation data (no classifier is trained)."""
    name = "embedding_similarity"

    def __init__(self, embedder):
        self.embedder = embedder
        self.proto = {c: embedder.encode(v) for c, v in PROTOTYPES.items()}

    def scores(self, texts) -> np.ndarray:
        e = self.embedder.encode(list(texts))
        return np.column_stack([(e @ self.proto[c].T).max(axis=1) for c in SPECIFIC])

    def fit(self, val_texts, y_val):
        s = self.scores(val_texts)
        grid = np.quantile(s, np.linspace(0.5, 0.995, 100))
        self.thresholds = np.array([grid[int(np.argmax([f1_score(y_val[:, j], s[:, j] >= t, zero_division=0) for t in grid]))]
                                    for j in range(len(SPECIFIC))])
        return self

    def predict(self, texts) -> np.ndarray:
        return (self.scores(texts) >= self.thresholds).astype(int)


class RuleModel:
    name = "keyword_rules"

    def predict(self, texts) -> np.ndarray:
        return np.array([[d[c] for c in SPECIFIC] for d in rules.label_many(texts)])


def per_category(y: np.ndarray, pred: np.ndarray) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(y, pred, zero_division=0)
    return [{"category": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(SPECIFIC)]


def summary(y, pred) -> dict:
    return {"micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def train_and_evaluate(clean: pd.DataFrame, labels: pd.DataFrame | None, embedder, out_dir: Path, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clean[clean["include"]].copy()
    if labels is not None:
        df = df.merge(labels, on="review_id", how="inner")
        label_source = "synthetic_generator_labels"
    else:
        weak = pd.DataFrame(rules.label_many(df["review_text"]))
        for c in SPECIFIC:
            df[c] = weak[c].to_numpy()
        df["split"] = np.where(df["review_id"].map(_bucket) < 0.8, "train", "test")
        label_source = "weak_keyword_labels (agreement only, not accuracy)"
    if labels is not None:
        df["role"] = df["split"]  # generator split: validation/test use disjoint unseen phrasings
    else:
        df["role"] = np.where(df["split"] == "test", "test", np.where(df["review_id"].map(_bucket) < 0.2, "validation", "train"))
    tr, va, te = (df[df.role == r] for r in ("train", "validation", "test"))
    Y = {k: d[SPECIFIC].to_numpy() for k, d in (("train", tr), ("validation", va), ("test", te))}

    tfidf = TfidfComplaintModel().fit(tr["review_text"], Y["train"], va["review_text"], Y["validation"])
    hybrid = HybridComplaintModel().fit(tr["review_text"], Y["train"], va["review_text"], Y["validation"])
    zs = EmbeddingZeroShotModel(embedder).fit(va["review_text"], Y["validation"])
    rule = RuleModel()
    models = {m.name: m for m in (rule, zs, tfidf, hybrid)}
    val_scores = {n: summary(Y["validation"], m.predict(va["review_text"])) for n, m in models.items()}
    best = max(val_scores, key=lambda n: val_scores[n]["micro_f1"])

    results = {}
    for n, m in models.items():
        pred = m.predict(te["review_text"])
        results[n] = {**summary(Y["test"], pred), "per_category": per_category(Y["test"], pred)}
    chosen = models[best]
    pred_te = chosen.predict(te["review_text"])
    errors = []
    for j, c in enumerate(SPECIFIC):
        fp = te[(pred_te[:, j] == 1) & (Y["test"][:, j] == 0)].head(2)
        fn = te[(pred_te[:, j] == 0) & (Y["test"][:, j] == 1)].head(2)
        errors += [{"category": c, "error": "false_positive", "review_id": r.review_id, "text": r.review_text} for r in fp.itertuples()]
        errors += [{"category": c, "error": "false_negative", "review_id": r.review_id, "text": r.review_text} for r in fn.itertuples()]

    # sentiment model: predicts the source recommendation from text
    sent = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", LogisticRegression(max_iter=2000, C=4.0))])
    sent.fit(tr["review_text"], (~tr["voted_up"].astype(bool)).astype(int))
    y_val_neg = (~va["voted_up"].astype(bool)).astype(int).to_numpy()
    sent_threshold = float(tune_thresholds(y_val_neg[:, None], sent.predict_proba(va["review_text"])[:, [1]])[0])
    p_neg = sent.predict_proba(te["review_text"])[:, 1]
    y_neg = (~te["voted_up"].astype(bool)).astype(int).to_numpy()
    cm = confusion_matrix(y_neg, (p_neg >= sent_threshold).astype(int), labels=[0, 1])
    pr, rc, f, _ = precision_recall_fscore_support(y_neg, p_neg >= sent_threshold, average="binary", zero_division=0)
    sentiment = {"target": "source recommendation (voted_up=False means negative)", "test_rows": int(len(te)),
                 "negative_rate": float(y_neg.mean()), "roc_auc": float(roc_auc_score(y_neg, p_neg)),
                 "threshold": sent_threshold,
                 "precision_negative": float(pr), "recall_negative": float(rc), "f1_negative": float(f),
                 "confusion_matrix": {"labels": ["positive", "negative"], "matrix": cm.tolist()},
                 "accuracy": float((cm[0, 0] + cm[1, 1]) / cm.sum())}

    class_dist = {c: int(df[c].sum()) for c in SPECIFIC}
    metrics = {"label_source": label_source, "rows": {k: int(len(v)) for k, v in (("train", tr), ("validation", va), ("test", te))},
               "selection": {"rule": "highest validation micro-F1", "validation": val_scores, "selected": best},
               "complaints_test": results, "sentiment_test": sentiment, "class_distribution": class_dist,
               "embedder": embedder.name,
               "caveat": ("Demo scores are measured against synthetic generator labels with held-out phrasing; they are "
                          "an optimistic proxy and do not establish accuracy on real Steam reviews.")}
    if isinstance(chosen, TfidfComplaintModel):
        metrics["top_terms"] = chosen.top_terms()
    (out_dir / "nlp_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "nlp_error_examples.json").write_text(json.dumps(errors, indent=2))
    joblib.dump({"complaint_model": chosen, "sentiment_model": sent, "sentiment_threshold": sent_threshold,
                 "selected": best, "label_source": label_source, "all_models": {"keyword_rules": rule, "tfidf_logreg": tfidf,
                 "tfidf_plus_rules": hybrid}}, out_dir / "nlp_models.joblib")
    return {"metrics": metrics, "selected_model": chosen, "sentiment_model": sent, "sentiment_threshold": sent_threshold}


def predict_all(clean: pd.DataFrame, complaint_model, sentiment_model, sentiment_threshold: float = 0.5) -> pd.DataFrame:
    df = clean[clean["include"]][["review_id"]].copy()
    texts = clean.loc[clean["include"], "review_text"]
    pred = complaint_model.predict(texts)
    for j, c in enumerate(SPECIFIC):
        df[c] = pred[:, j].astype(bool)
    neg = ~clean.loc[clean["include"], "voted_up"].astype(bool).to_numpy()
    df["other"] = neg & ~pred.any(axis=1).astype(bool)
    df["model_p_negative"] = sentiment_model.predict_proba(texts)[:, 1]
    df["model_sentiment"] = np.where(df["model_p_negative"] >= sentiment_threshold, "negative", "positive")
    return df
