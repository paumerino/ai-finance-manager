"""Hybrid rule-based and ML expense/income categorizer using scikit-learn."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

KEYWORD_RULES: list[tuple[str, str, list[str]]] = [
    (
        "Income",
        "income",
        ["salary", "payroll", "stipend", "grant", "refund", "invoice paid", "nomina", "beca"],
    ),
    (
        "Housing & Utilities",
        "expense",
        [
            "rent", "alquiler", "mortgage", "hipoteca", "electricity", "iberdrola",
            "endesa", "water", "agua", "gas", "internet", "fiber", "fibra", "utilities",
        ],
    ),
    (
        "Groceries",
        "expense",
        [
            "mercadona", "lidl", "carrefour", "aldi", "dia", "consum", "bonpreu",
            "biedronka", "zabka", "supermarket", "supermercado", "grocery", "groceries",
        ],
    ),
    (
        "Transport",
        "expense",
        [
            "uber", "cabify", "bolt", "renfe", "metro", "bus", "train", "ticket",
            "ztm", "mpk", "parking", "fuel", "gasolina", "petrol", "flight", "ryanair", "vueling",
        ],
    ),
    (
        "Dining & Leisure",
        "expense",
        [
            "restaurant", "restaurante", "cafe", "coffee", "bar", "pub", "cinema",
            "cinesa", "pizza", "pizzeria", "burger", "mcdonald", "kfc", "dinner", "lunch",
        ],
    ),
    (
        "Subscriptions & Tech",
        "expense",
        [
            "spotify", "netflix", "youtube", "github", "chatgpt", "openai", "icloud",
            "google storage", "cloud storage", "adobe", "notion", "steam", "subscription",
        ],
    ),
    (
        "Health & Fitness",
        "expense",
        [
            "gym", "fitness", "pharmacy", "farmacia", "clinic", "clinica",
            "doctor", "hospital", "dentist", "health", "salud",
        ],
    ),
]


@dataclass
class CategorizationPrediction:
    category: str
    source: str  # "rule" | "ml" | "fallback"
    confidence: float


@dataclass
class CategorizerEvaluation:
    samples_evaluated: int
    train_samples: int
    test_samples: int
    accuracy: float
    macro_f1: float


def _normalize_text(merchant: str, description: str = "") -> str:
    combined = f"{merchant} {description}".lower().strip()
    combined = re.sub(r"[^a-z0-9áéíóúüñ\s/+-]", " ", combined)
    return re.sub(r"\s+", " ", combined).strip()


def predict_by_rule(
    merchant: str,
    description: str = "",
    tx_type: str | None = None,
) -> str | None:
    """Return a category if a deterministic keyword rule matches."""
    text = _normalize_text(merchant, description)
    if not text:
        return None

    for category, rule_type, keywords in KEYWORD_RULES:
        if tx_type and tx_type != rule_type:
            continue
        for kw in keywords:
            pattern = rf"(?<!\w){re.escape(kw)}(?!\w)"
            if re.search(pattern, text):
                return category

    if tx_type == "income":
        return "Income"
    return None


class ExpenseCategorizer:
    """Supervised ML categorizer with TF-IDF word+char features and rule fallback."""

    def __init__(self, min_samples_to_train: int = 12) -> None:
        self.min_samples_to_train = min_samples_to_train
        self.pipeline: Pipeline | None = None
        self.is_fitted: bool = False
        self.classes_: list[str] = []

    def _build_pipeline(self) -> Pipeline:
        features = FeatureUnion(
            [
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        lowercase=True,
                        min_df=1,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        lowercase=True,
                        min_df=1,
                    ),
                ),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    def fit(self, records: Iterable[tuple[str, str, str]]) -> bool:
        """Train on (merchant, description, category) tuples. Returns True if ML fitted."""
        rows = [
            (_normalize_text(m, d), cat.strip())
            for m, d, cat in records
            if _normalize_text(m, d) and cat and cat.strip() and cat.strip() != "Uncategorized"
        ]
        if len(rows) < self.min_samples_to_train:
            self.is_fitted = False
            self.pipeline = None
            self.classes_ = []
            return False

        texts = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            self.is_fitted = False
            self.pipeline = None
            self.classes_ = unique_labels
            return False

        pipe = self._build_pipeline()
        pipe.fit(texts, labels)
        self.pipeline = pipe
        self.is_fitted = True
        self.classes_ = list(pipe.classes_)
        return True

    def fit_from_dataframe(self, df: pd.DataFrame) -> bool:
        """Fit from a transactions DataFrame containing merchant, description, and category."""
        if df.empty or "merchant" not in df.columns or "category" not in df.columns:
            self.is_fitted = False
            return False

        descriptions = df["description"] if "description" in df.columns else [""] * len(df)
        records = zip(df["merchant"].astype(str), descriptions.astype(str), df["category"].astype(str))
        return self.fit(records)

    def predict_one(
        self,
        merchant: str,
        description: str = "",
        tx_type: str | None = None,
        prefer_rules: bool = True,
    ) -> CategorizationPrediction:
        """Predict category for a single transaction using rules + ML + fallback."""
        if prefer_rules:
            rule_cat = predict_by_rule(merchant, description, tx_type=tx_type)
            if rule_cat:
                return CategorizationPrediction(category=rule_cat, source="rule", confidence=0.95)

        text = _normalize_text(merchant, description)
        if self.is_fitted and self.pipeline is not None and text:
            probs = self.pipeline.predict_proba([text])[0]
            best_idx = int(probs.argmax())
            pred_cat = str(self.pipeline.classes_[best_idx])
            confidence = round(float(probs[best_idx]), 3)
            if tx_type == "income" and pred_cat != "Income" and "Income" in self.classes_:
                return CategorizationPrediction(category="Income", source="ml", confidence=confidence)
            return CategorizationPrediction(category=pred_cat, source="ml", confidence=confidence)

        if not prefer_rules:
            rule_cat = predict_by_rule(merchant, description, tx_type=tx_type)
            if rule_cat:
                return CategorizationPrediction(category=rule_cat, source="rule", confidence=0.9)

        fallback_cat = "Income" if tx_type == "income" else "Shopping & Misc"
        return CategorizationPrediction(category=fallback_cat, source="fallback", confidence=0.3)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add predicted_category, prediction_source, and confidence columns to a DataFrame."""
        out = df.copy()
        if out.empty:
            out["predicted_category"] = pd.Series(dtype=str)
            out["prediction_source"] = pd.Series(dtype=str)
            out["confidence"] = pd.Series(dtype=float)
            return out

        descriptions = out["description"] if "description" in out.columns else [""] * len(out)
        tx_types = out["tx_type"] if "tx_type" in out.columns else [None] * len(out)

        preds = [
            self.predict_one(str(m), str(d), tx_type=str(t) if t else None)
            for m, d, t in zip(out["merchant"], descriptions, tx_types)
        ]
        out["predicted_category"] = [p.category for p in preds]
        out["prediction_source"] = [p.source for p in preds]
        out["confidence"] = [p.confidence for p in preds]
        return out


def evaluate_ml_categorizer(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> CategorizerEvaluation | None:
    """Evaluate pure ML classification (without keyword shortcuts) on a labeled DataFrame."""
    if df.empty or "merchant" not in df.columns or "category" not in df.columns:
        return None

    descriptions = df["description"] if "description" in df.columns else pd.Series([""] * len(df))
    clean_df = pd.DataFrame(
        {
            "text": [_normalize_text(str(m), str(d)) for m, d in zip(df["merchant"], descriptions)],
            "category": df["category"].astype(str).str.strip(),
        }
    )
    clean_df = clean_df[
        (clean_df["text"] != "")
        & (clean_df["category"] != "")
        & (clean_df["category"] != "Uncategorized")
    ]

    if len(clean_df) < 16 or clean_df["category"].nunique() < 2:
        return None

    min_class_count = clean_df["category"].value_counts().min()
    stratify_arg = clean_df["category"] if min_class_count >= 2 else None

    train_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    model = ExpenseCategorizer(min_samples_to_train=4)
    fitted = model.fit([(txt, "", cat) for txt, cat in zip(train_df["text"], train_df["category"])])
    if not fitted or model.pipeline is None:
        return None

    y_true = test_df["category"].tolist()
    y_pred = model.pipeline.predict(test_df["text"].tolist()).tolist()

    return CategorizerEvaluation(
        samples_evaluated=len(clean_df),
        train_samples=len(train_df),
        test_samples=len(test_df),
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        macro_f1=round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    )


if __name__ == "__main__":
    from src.database import DEFAULT_DB_PATH, get_transactions_df

    tx_df = get_transactions_df(DEFAULT_DB_PATH)
    clf = ExpenseCategorizer()
    fitted = clf.fit_from_dataframe(tx_df)
    metrics = evaluate_ml_categorizer(tx_df)

    print(f"ML model fitted: {fitted} | Classes: {clf.classes_}")
    if metrics:
        print(
            f"ML Holdout Eval ({metrics.test_samples} test items): "
            f"Accuracy={metrics.accuracy * 100:.1f}%, Macro-F1={metrics.macro_f1 * 100:.1f}%"
        )

    samples = [
        ("MERCADONA TARRAGONA", "", "expense"),
        ("UBER EATS / UBER TRIP", "", "expense"),
        ("SPOTIFY EUROPE", "", "expense"),
        ("MONTHLY PAYROLL", "", "income"),
        ("LOCAL CORNER SHOP", "", "expense"),
    ]
    for merchant, desc, tx_type in samples:
        res = clf.predict_one(merchant, desc, tx_type=tx_type)
        print(f"  - {merchant:<24} -> {res.category:<22} ({res.source}, conf={res.confidence:.2f})")