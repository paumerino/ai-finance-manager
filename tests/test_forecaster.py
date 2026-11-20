"""Unit tests for monthly expense forecasting, walk-forward validation, and anomalies."""

import pandas as pd

from src.forecaster import detect_expense_anomalies, forecast_monthly_expenses


def _sample_series_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    months = [
        ("2026-05", 820.0),
        ("2026-06", 850.0),
        ("2026-07", 830.0),
        ("2026-08", 870.0),
        ("2026-09", 840.0),
        ("2026-10", 860.0),
    ]
    tx_id = 1
    for ym, total_exp in months:
        rows.append(
            {
                "id": tx_id,
                "tx_date": f"{ym}-02",
                "year_month": ym,
                "merchant": "RENT",
                "amount": 550.0,
                "tx_type": "expense",
                "category": "Housing & Utilities",
            }
        )
        tx_id += 1
        rows.append(
            {
                "id": tx_id,
                "tx_date": f"{ym}-10",
                "year_month": ym,
                "merchant": "SUPERMARKET",
                "amount": round(total_exp - 550.0, 2),
                "tx_type": "expense",
                "category": "Groceries",
            }
        )
        tx_id += 1
    return pd.DataFrame(rows)


def test_forecast_produces_next_month_and_walk_forward_metrics() -> None:
    df = _sample_series_df()
    summary = forecast_monthly_expenses(df)

    assert summary.target_month == "2026-11"
    assert summary.historical_months == 6
    assert summary.predicted_total_expense > 0
    assert summary.lower_bound <= summary.predicted_total_expense <= summary.upper_bound
    assert summary.walk_forward_mae is not None
    assert summary.walk_forward_mape is not None
    assert "Housing & Utilities" in summary.by_category
    assert "Groceries" in summary.by_category


def test_detect_expense_anomalies_flags_outlier_spike() -> None:
    df = _sample_series_df()
    # Add 4 normal small shopping transactions + 1 big spike
    for idx, amt in enumerate([20.0, 25.0, 22.0, 28.0], start=100):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "id": idx,
                            "tx_date": "2026-09-15",
                            "year_month": "2026-09",
                            "merchant": "BOOKSTORE",
                            "amount": amt,
                            "tx_type": "expense",
                            "category": "Shopping & Misc",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    spike = pd.DataFrame(
        [
            {
                "id": 999,
                "tx_date": "2026-10-18",
                "year_month": "2026-10",
                "merchant": "LAPTOP REPAIR",
                "amount": 350.0,
                "tx_type": "expense",
                "category": "Shopping & Misc",
            }
        ]
    )
    df = pd.concat([df, spike], ignore_index=True)

    anomalies = detect_expense_anomalies(df)
    assert len(anomalies) == 1
    assert anomalies[0].merchant == "LAPTOP REPAIR"
    assert anomalies[0].severity == "high"