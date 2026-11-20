"""Monthly expense forecasting with walk-forward validation and anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


@dataclass
class ForecastSummary:
    target_month: str
    predicted_total_expense: float
    lower_bound: float
    upper_bound: float
    method: str
    historical_months: int
    walk_forward_mae: float | None
    walk_forward_mape: float | None
    by_category: dict[str, float]


@dataclass
class ExpenseAnomaly:
    tx_id: int | None
    tx_date: str
    merchant: str
    category: str
    amount: float
    category_median: float
    severity: str  # "moderate" | "high"
    reason: str


def _next_year_month(last_ym: str) -> str:
    """Given 'YYYY-MM', return the next month as 'YYYY-MM'."""
    year_str, month_str = last_ym.split("-")
    year, month = int(year_str), int(month_str)
    month += 1
    if month > 12:
        month = 1
        year += 1
    return f"{year:04d}-{month:02d}"


def _build_time_features(ym_list: list[str], start_index: int = 0) -> np.ndarray:
    """Build deterministic trend + cyclical month features for a sequence of YYYY-MM strings."""
    rows: list[list[float]] = []
    for idx, ym in enumerate(ym_list):
        month = int(ym.split("-")[1])
        t = float(start_index + idx)
        angle = 2.0 * math.pi * (month / 12.0)
        rows.append([t, math.sin(angle), math.cos(angle)])
    return np.asarray(rows, dtype=float)


def _predict_series_next(values: list[float], months_ym: list[str]) -> tuple[float, str]:
    """Predict next value for a monthly series using Ridge regression or damped mean."""
    n = len(values)
    if n == 0:
        return 0.0, "empty"
    if n == 1:
        return round(max(0.0, float(values[0])), 2), "single_month_baseline"
    if n < 4:
        # Short history: blend recent average with gentle trend, clipped safely
        recent_avg = float(np.mean(values[-min(3, n) :]))
        diff = float(values[-1] - values[-2]) * 0.35
        pred = max(0.0, recent_avg + diff)
        return round(pred, 2), "damped_short_series"

    X_train = _build_time_features(months_ym, start_index=0)
    y_train = np.asarray(values, dtype=float)
    model = Ridge(alpha=2.5)
    model.fit(X_train, y_train)

    next_ym = _next_year_month(months_ym[-1])
    X_next = _build_time_features([next_ym], start_index=n)
    raw_pred = float(model.predict(X_next)[0])

    # Shrink slightly toward trailing 3-month mean to prevent wild extrapolation on noisy data
    trailing_mean = float(np.mean(y_train[-3:]))
    blended = 0.65 * raw_pred + 0.35 * trailing_mean
    return round(max(0.0, blended), 2), "ridge_time_features"


def evaluate_walk_forward(
    months_ym: list[str],
    values: list[float],
    min_train_months: int = 3,
) -> tuple[float | None, float | None]:
    """Evaluate 1-step-ahead predictions strictly without future leakage."""
    if len(values) <= min_train_months or len(values) != len(months_ym):
        return None, None

    actuals: list[float] = []
    preds: list[float] = []

    for split_idx in range(min_train_months, len(values)):
        history_vals = values[:split_idx]
        history_ym = months_ym[:split_idx]
        pred_val, _ = _predict_series_next(history_vals, history_ym)
        actuals.append(float(values[split_idx]))
        preds.append(float(pred_val))

    mae = float(mean_absolute_error(actuals, preds))
    non_zero_pairs = [(a, p) for a, p in zip(actuals, preds) if a > 0]
    mape = (
        float(np.mean([abs(a - p) / a for a, p in non_zero_pairs]) * 100.0)
        if non_zero_pairs
        else None
    )
    return round(mae, 2), round(mape, 1) if mape is not None else None


def forecast_monthly_expenses(
    transactions_df: pd.DataFrame,
    category_budgets: dict[str, float] | None = None,
) -> ForecastSummary:
    """Forecast total and per-category expenses for the next calendar month."""
    if transactions_df.empty:
        return ForecastSummary(
            target_month="N/A",
            predicted_total_expense=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            method="no_data",
            historical_months=0,
            walk_forward_mae=None,
            walk_forward_mape=None,
            by_category={},
        )

    df = transactions_df.copy()
    if "year_month" not in df.columns:
        df["year_month"] = df["tx_date"].astype(str).str.slice(0, 7)

    expenses_df = df[df["tx_type"] == "expense"]
    if expenses_df.empty:
        last_ym = sorted(df["year_month"].astype(str).unique())[-1]
        return ForecastSummary(
            target_month=_next_year_month(last_ym),
            predicted_total_expense=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            method="no_expenses",
            historical_months=0,
            walk_forward_mae=None,
            walk_forward_mape=None,
            by_category={},
        )

    monthly_totals = (
        expenses_df.groupby("year_month", as_index=False)["amount"]
        .sum()
        .sort_values("year_month")
    )
    months_ym = monthly_totals["year_month"].astype(str).tolist()
    total_series = monthly_totals["amount"].astype(float).tolist()

    target_month = _next_year_month(months_ym[-1])
    pred_total, method = _predict_series_next(total_series, months_ym)
    mae, mape = evaluate_walk_forward(months_ym, total_series, min_train_months=3)

    # Confidence band around prediction using walk-forward MAE or ~15% heuristic fallback
    band = mae if mae is not None and mae > 0 else round(pred_total * 0.15, 2)
    lower_bound = round(max(0.0, pred_total - band), 2)
    upper_bound = round(pred_total + band, 2)

    # Forecast by category
    by_category: dict[str, float] = {}
    cat_pivot = (
        expenses_df.pivot_table(
            index="year_month",
            columns="category",
            values="amount",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(months_ym, fill_value=0.0)
    )

    for cat in cat_pivot.columns:
        cat_vals = cat_pivot[cat].astype(float).tolist()
        cat_pred, _ = _predict_series_next(cat_vals, months_ym)
        if cat_pred > 0:
            by_category[str(cat)] = cat_pred

    # Include tracked categories with budget even if historical spend in series was 0
    if category_budgets:
        for cat_name, budget_val in category_budgets.items():
            if cat_name not in by_category and budget_val > 0:
                by_category[cat_name] = 0.0

    by_category = dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True))

    return ForecastSummary(
        target_month=target_month,
        predicted_total_expense=pred_total,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        method=method,
        historical_months=len(months_ym),
        walk_forward_mae=mae,
        walk_forward_mape=mape,
        by_category=by_category,
    )


def detect_expense_anomalies(
    transactions_df: pd.DataFrame,
    min_category_txs: int = 4,
    multiplier_threshold: float = 2.5,
    min_excess_amount: float = 40.0,
) -> list[ExpenseAnomaly]:
    """Flag unusually large expense transactions relative to category median/MAD."""
    if transactions_df.empty:
        return []

    expenses = transactions_df[transactions_df["tx_type"] == "expense"].copy()
    if expenses.empty:
        return []

    anomalies: list[ExpenseAnomaly] = []

    for category, grp in expenses.groupby("category"):
        if len(grp) < min_category_txs:
            continue

        amounts = grp["amount"].astype(float)
        median_val = float(amounts.median())
        if median_val <= 0:
            continue

        mad = float((amounts - median_val).abs().median())
        # Scale MAD to normal std equivalent; fall back to IQR/median when MAD is zero (e.g., fixed bills)
        robust_scale = 1.4826 * mad if mad > 0 else max(median_val * 0.25, 1.0)

        for _, row in grp.iterrows():
            amt = float(row["amount"])
            excess = amt - median_val
            robust_z = (amt - median_val) / robust_scale

            if amt > median_val * multiplier_threshold and excess >= min_excess_amount and robust_z >= 2.5:
                severity = "high" if (amt >= median_val * 4.0 or robust_z >= 5.0) else "moderate"
                ratio = amt / median_val
                anomalies.append(
                    ExpenseAnomaly(
                        tx_id=int(row["id"]) if "id" in row and pd.notna(row["id"]) else None,
                        tx_date=str(row["tx_date"]),
                        merchant=str(row["merchant"]),
                        category=str(category),
                        amount=round(amt, 2),
                        category_median=round(median_val, 2),
                        severity=severity,
                        reason=(
                            f"{amt:.2f} is {ratio:.1f}x the {category} median ({median_val:.2f})."
                        ),
                    )
                )

    anomalies.sort(key=lambda a: a.amount, reverse=True)
    return anomalies


if __name__ == "__main__":
    from src.database import DEFAULT_DB_PATH, get_transactions_df

    tx_df = get_transactions_df(DEFAULT_DB_PATH)
    summary = forecast_monthly_expenses(tx_df)
    found_anomalies = detect_expense_anomalies(tx_df)

    print(f"\n=== EXPENSE FORECAST FOR {summary.target_month} ===")
    print(f"Predicted total: {summary.predicted_total_expense:.2f} (range: {summary.lower_bound:.2f} - {summary.upper_bound:.2f})")
    print(f"Method: {summary.method} | History: {summary.historical_months} months")
    print(f"Walk-forward MAE: {summary.walk_forward_mae} | MAPE: {summary.walk_forward_mape}%")

    print("\nPredicted by category:")
    for cat, val in summary.by_category.items():
        print(f"  - {cat:<22}: {val:.2f}")

    print(f"\nDetected anomalies ({len(found_anomalies)}):")
    for anom in found_anomalies[:8]:
        print(f"  [{anom.severity.upper()}] {anom.tx_date} | {anom.merchant} | {anom.amount:.2f} ({anom.reason})")