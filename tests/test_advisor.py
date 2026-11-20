"""Unit tests for the structured AI financial advisor and deterministic fallback."""

import pandas as pd

from src.advisor import (
    FinanceAdvisorOutput,
    FinanceAdvisorPayload,
    _build_finance_prompt,
    generate_finance_advice,
)
from src.forecaster import ExpenseAnomaly, ForecastSummary


def _sample_inputs():
    monthly_df = pd.DataFrame(
        [
            {
                "year_month": "2026-10",
                "income": 1200.0,
                "expenses": 890.0,
                "net_savings": 310.0,
                "savings_rate": 25.8,
                "tx_count": 18,
            }
        ]
    )
    breakdown_df = pd.DataFrame(
        [
            {
                "category": "Housing & Utilities",
                "monthly_budget": 600.0,
                "spent": 570.0,
                "remaining": 30.0,
                "budget_usage_pct": 95.0,
            },
            {
                "category": "Groceries",
                "monthly_budget": 200.0,
                "spent": 220.0,
                "remaining": -20.0,
                "budget_usage_pct": 110.0,
            },
        ]
    )
    forecast = ForecastSummary(
        target_month="2026-11",
        predicted_total_expense=910.0,
        lower_bound=820.0,
        upper_bound=1000.0,
        method="ridge_time_features",
        historical_months=6,
        walk_forward_mae=45.0,
        walk_forward_mape=5.2,
        by_category={"Housing & Utilities": 570.0, "Groceries": 210.0},
    )
    anomalies = [
        ExpenseAnomaly(
            tx_id=10,
            tx_date="2026-10-15",
            merchant="LAPTOP REPAIR",
            category="Shopping & Misc",
            amount=340.0,
            category_median=35.0,
            severity="high",
            reason="340.00 is 9.7x the Shopping & Misc median (35.00).",
        )
    ]
    return monthly_df, breakdown_df, forecast, anomalies


def test_finance_advisor_fallback_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monthly_df, breakdown_df, forecast, anomalies = _sample_inputs()

    out = generate_finance_advice(monthly_df, breakdown_df, forecast, anomalies, api_key="")
    assert isinstance(out, FinanceAdvisorOutput)
    assert out.source == "rule_based_fallback"
    assert len(out.category_recommendations) >= 1
    assert len(out.next_month_focus) == 3


def test_finance_prompt_includes_sql_and_forecast_data() -> None:
    monthly_df, breakdown_df, forecast, anomalies = _sample_inputs()
    prompt = _build_finance_prompt(monthly_df, breakdown_df, forecast, anomalies, selected_month="2026-10")

    assert "2026-11" in prompt
    assert "910.00" in prompt
    assert "LAPTOP REPAIR" in prompt


def test_finance_pydantic_schema_validation() -> None:
    raw_json = """
    {
      "monthly_health_summary": "Solid savings rate last month with a minor grocery budget overrun.",
      "anomaly_insights": ["LAPTOP REPAIR at 340.00 was an outlier."],
      "category_recommendations": [
        {
          "category": "Groceries",
          "suggested_action": "Plan weekly meals before shopping.",
          "estimated_monthly_impact": "~25 EUR/month"
        }
      ],
      "next_month_focus": ["Track Groceries weekly", "Keep rent steady", "Set aside savings early"]
    }
    """
    parsed = FinanceAdvisorPayload.model_validate_json(raw_json)
    assert len(parsed.category_recommendations) == 1
    assert len(parsed.next_month_focus) == 3