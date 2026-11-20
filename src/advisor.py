"""Grounded AI financial advisor using Gemini structured JSON with deterministic fallback."""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.forecaster import ExpenseAnomaly, ForecastSummary

DEFAULT_LLM_MODEL = "gemini-3.8-flash"


class CategoryAdjustment(BaseModel):
    category: str = Field(description="Expense category name.")
    suggested_action: str = Field(description="Concrete, realistic saving or budget tip.")
    estimated_monthly_impact: str = Field(
        description="Realistic monthly saving estimate or budget adjustment note."
    )


class FinanceAdvisorPayload(BaseModel):
    """Strict JSON schema enforced on the LLM response."""

    monthly_health_summary: str = Field(
        description="2-3 sentence honest summary of income, spending, savings rate, and forecast."
    )
    anomaly_insights: list[str] = Field(
        description="Observations on unusual transactions or spending spikes (or why spending is steady)."
    )
    category_recommendations: list[CategoryAdjustment] = Field(
        description="2-4 actionable recommendations grounded in actual spending and budgets."
    )
    next_month_focus: list[str] = Field(
        description="3 concrete steps for the upcoming month."
    )


class FinanceAdvisorOutput(FinanceAdvisorPayload):
    source: str = "rule_based_fallback"  # "llm" | "rule_based_fallback"
    model_used: str = "deterministic-fallback"
    fallback_reason: str | None = None


def _build_finance_prompt(
    monthly_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    forecast: ForecastSummary,
    anomalies: list[ExpenseAnomaly],
    selected_month: str | None = None,
) -> str:
    """Ground the LLM in exact SQL aggregates, forecast numbers, and anomaly records."""
    recent_months = (
        monthly_df.tail(6).to_string(index=False) if not monthly_df.empty else "No monthly data."
    )
    cat_summary = (
        breakdown_df[["category", "spent", "monthly_budget", "budget_usage_pct"]].to_string(
            index=False
        )
        if not breakdown_df.empty
        else "No category data."
    )
    anom_lines = (
        "\n".join(
            f"- {a.tx_date} | {a.merchant} | {a.category} | {a.amount:.2f} | {a.reason}"
            for a in anomalies[:6]
        )
        if anomalies
        else "- No major transaction anomalies flagged."
    )

    return f"""
You are a pragmatic personal finance analyst and data-driven coach.
Provide practical, non-judgmental financial guidance grounded strictly in the metrics below.

RULES:
- Do NOT invent transactions, balances, or numbers not present in the data.
- Keep recommendations realistic for a student/early-career budget.
- Be concise, clear, and actionable (English).

TARGET FOCUS MONTH: {selected_month or "Latest available"}

RECENT MONTHLY SUMMARY (SQL):
{recent_months}

CATEGORY SPENDING VS BUDGET:
{cat_summary}

NEXT MONTH FORECAST ({forecast.target_month}):
- Predicted total expense: {forecast.predicted_total_expense:.2f} (range {forecast.lower_bound:.2f} - {forecast.upper_bound:.2f})
- Method: {forecast.method} | Walk-forward MAE: {forecast.walk_forward_mae}

FLAGGED ANOMALIES:
{anom_lines}
""".strip()


def generate_rule_based_finance_advice(
    monthly_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    forecast: ForecastSummary,
    anomalies: list[ExpenseAnomaly],
    reason: str | None = None,
) -> FinanceAdvisorOutput:
    """Deterministic fallback so the app works cleanly without an API key."""
    if not monthly_df.empty:
        latest = monthly_df.iloc[-1]
        ym = str(latest["year_month"])
        inc = float(latest["income"])
        exp = float(latest["expenses"])
        sav_rate = float(latest["savings_rate"])
        summary = (
            f"In {ym}, recorded income was {inc:.2f} and expenses were {exp:.2f} "
            f"(savings rate: {sav_rate:.1f}%). Forecast for {forecast.target_month} is "
            f"{forecast.predicted_total_expense:.2f} in expenses."
        )
    else:
        summary = (
            f"Forecast for {forecast.target_month} estimates {forecast.predicted_total_expense:.2f} "
            "in expenses based on available transactions."
        )

    if anomalies:
        anom_insights = [
            f"{a.tx_date} ({a.merchant}, {a.category}): {a.amount:.2f} — {a.reason}"
            for a in anomalies[:3]
        ]
    else:
        anom_insights = ["No unusual spending spikes exceeded the anomaly threshold."]

    recs: list[CategoryAdjustment] = []
    if not breakdown_df.empty:
        over_or_high = breakdown_df[breakdown_df["spent"] > 0].sort_values("spent", ascending=False)
        for _, row in over_or_high.head(3).iterrows():
            cat_name = str(row["category"])
            spent_val = float(row["spent"])
            budget_val = float(row["monthly_budget"])
            if budget_val > 0 and spent_val > budget_val:
                action = (
                    f"Spending ({spent_val:.2f}) exceeded the {budget_val:.2f} budget; "
                    "cap weekly discretionary orders and track recurring items."
                )
                impact = f"Target bringing spend closer to {budget_val:.2f} budget."
            else:
                trim = round(spent_val * 0.10, 2)
                action = (
                    f"Review recurring and impulse items in {cat_name} (current spend: {spent_val:.2f})."
                )
                impact = f"~{trim:.2f}/month potential saving with a 10% trim."
            recs.append(
                CategoryAdjustment(
                    category=cat_name,
                    suggested_action=action,
                    estimated_monthly_impact=impact,
                )
            )

    if not recs:
        recs.append(
            CategoryAdjustment(
                category="General Budget",
                suggested_action="Log all weekly expenses and review category budgets monthly.",
                estimated_monthly_impact="Improves visibility and savings consistency.",
            )
        )

    focus = [
        f"Set a clear spending guard around {forecast.predicted_total_expense:.2f} for {forecast.target_month}.",
        "Verify ML-categorized transactions and fix any misclassified merchants.",
        "Review flagged outliers to separate one-off necessities from avoidable spikes.",
    ]

    return FinanceAdvisorOutput(
        monthly_health_summary=summary,
        anomaly_insights=anom_insights,
        category_recommendations=recs,
        next_month_focus=focus,
        source="rule_based_fallback",
        model_used="deterministic-fallback",
        fallback_reason=reason,
    )


def generate_finance_advice(
    monthly_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    forecast: ForecastSummary,
    anomalies: list[ExpenseAnomaly],
    selected_month: str | None = None,
    api_key: str | None = None,
    model_name: str = DEFAULT_LLM_MODEL,
    use_llm: bool = True,
) -> FinanceAdvisorOutput:
    """Generate structured financial advice via Gemini with deterministic fallback."""
    load_dotenv()
    resolved_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    if not use_llm:
        return generate_rule_based_finance_advice(
            monthly_df, breakdown_df, forecast, anomalies, reason="AI Advisor toggle is off."
        )

    if not resolved_key or resolved_key == "your_api_key_here":
        return generate_rule_based_finance_advice(
            monthly_df,
            breakdown_df,
            forecast,
            anomalies,
            reason="GEMINI_API_KEY not configured; using deterministic fallback.",
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)
        prompt = _build_finance_prompt(
            monthly_df=monthly_df,
            breakdown_df=breakdown_df,
            forecast=forecast,
            anomalies=anomalies,
            selected_month=selected_month,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinanceAdvisorPayload,
                temperature=0.3,
            ),
        )

        payload = FinanceAdvisorPayload.model_validate_json(response.text)
        return FinanceAdvisorOutput(
            **payload.model_dump(),
            source="llm",
            model_used=model_name,
            fallback_reason=None,
        )
    except Exception as exc:
        return generate_rule_based_finance_advice(
            monthly_df,
            breakdown_df,
            forecast,
            anomalies,
            reason=f"LLM call failed ({type(exc).__name__}); using deterministic fallback.",
        )


if __name__ == "__main__":
    from src.database import DEFAULT_DB_PATH, get_category_breakdown_df, get_monthly_summary_df, get_transactions_df
    from src.forecaster import detect_expense_anomalies, forecast_monthly_expenses

    tx_df = get_transactions_df(DEFAULT_DB_PATH)
    monthly = get_monthly_summary_df(DEFAULT_DB_PATH)
    latest_m = str(monthly.iloc[-1]["year_month"]) if not monthly.empty else None
    breakdown = get_category_breakdown_df(DEFAULT_DB_PATH, month=latest_m)
    fc = forecast_monthly_expenses(tx_df)
    anoms = detect_expense_anomalies(tx_df)

    out = generate_finance_advice(monthly, breakdown, fc, anoms, selected_month=latest_m)
    print(f"\n=== FINANCE ADVISOR ({out.source} | {out.model_used}) ===")
    if out.fallback_reason:
        print(f"Note: {out.fallback_reason}")
    print(f"\nSummary:\n{out.monthly_health_summary}")
    print("\nAnomaly Insights:")
    for item in out.anomaly_insights:
        print(f"  - {item}")
    print("\nCategory Recommendations:")
    for r in out.category_recommendations:
        print(f"  [{r.category}] {r.suggested_action} ({r.estimated_monthly_impact})")
    print("\nNext Month Focus:")
    for idx, step in enumerate(out.next_month_focus, start=1):
        print(f"  {idx}. {step}")