"""Unit tests for rule-based and ML transaction categorization."""

import pandas as pd

from src.categorizer import (
    ExpenseCategorizer,
    evaluate_ml_categorizer,
    predict_by_rule,
)


def test_rule_categorizer_detects_known_merchants() -> None:
    assert predict_by_rule("MERCADONA SUPERMERCADO", tx_type="expense") == "Groceries"
    assert predict_by_rule("RENFE CERCANIAS", tx_type="expense") == "Transport"
    assert predict_by_rule("SPOTIFY AB", tx_type="expense") == "Subscriptions & Tech"
    assert predict_by_rule("UNIVERSITY STIPEND", tx_type="income") == "Income"


def test_categorizer_fallback_when_untrained() -> None:
    clf = ExpenseCategorizer(min_samples_to_train=20)
    assert clf.fit([("Mercadona", "", "Groceries")]) is False

    pred = clf.predict_one("UNKNOWN SHOP", tx_type="expense")
    assert pred.category == "Shopping & Misc"
    assert pred.source == "fallback"


def test_ml_categorizer_trains_and_evaluates() -> None:
    rows = [
        ("MERCADONA REUS", "food", "Groceries"),
        ("LIDL MARKET", "food", "Groceries"),
        ("CARREFOUR EXPRESS", "food", "Groceries"),
        ("ALDI SUPERMERCADO", "food", "Groceries"),
        ("UBER TRIP", "ride", "Transport"),
        ("RENFE TRAIN", "commute", "Transport"),
        ("METRO TICKET", "commute", "Transport"),
        ("CABIFY RIDE", "ride", "Transport"),
        ("SPOTIFY PLAN", "music", "Subscriptions & Tech"),
        ("NETFLIX STREAM", "video", "Subscriptions & Tech"),
        ("GITHUB PRO", "dev", "Subscriptions & Tech"),
        ("OPENAI API", "ai", "Subscriptions & Tech"),
        ("CAFE CENTRAL", "coffee", "Dining & Leisure"),
        ("PIZZERIA NAPOLI", "food", "Dining & Leisure"),
        ("CINESA TICKET", "movie", "Dining & Leisure"),
        ("RESTAURANTE PUERTO", "dinner", "Dining & Leisure"),
    ]
    df = pd.DataFrame(rows, columns=["merchant", "description", "category"])

    clf = ExpenseCategorizer(min_samples_to_train=8)
    assert clf.fit_from_dataframe(df) is True

    metrics = evaluate_ml_categorizer(df, test_size=0.25, random_state=42)
    assert metrics is not None
    assert metrics.samples_evaluated == 16
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0