"""Interactive Streamlit dashboard for the AI Personal Finance Manager."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.advisor import DEFAULT_LLM_MODEL, generate_finance_advice
from src.categorizer import ExpenseCategorizer, evaluate_ml_categorizer
from src.database import (
    DEFAULT_DB_PATH,
    add_transaction,
    get_category_breakdown_df,
    get_categories_df,
    get_monthly_summary_df,
    get_transactions_df,
    import_transactions_csv,
    init_db,
    update_category_budget,
    update_transaction_category,
)
from src.forecaster import detect_expense_anomalies, forecast_monthly_expenses
from src.sample_data import SAMPLE_CSV_PATH, export_sample_csv, seed_sample_data

load_dotenv()

st.set_page_config(
    page_title="AI Personal Finance Manager",
    page_icon="💶",
    layout="wide",
)


def _get_api_key(override_key: str) -> str:
    if override_key.strip():
        return override_key.strip()
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def _build_cashflow_chart(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly_df["year_month"],
            y=monthly_df["income"],
            name="Income",
            marker_color="#16a34a",
        )
    )
    fig.add_trace(
        go.Bar(
            x=monthly_df["year_month"],
            y=monthly_df["expenses"],
            name="Expenses",
            marker_color="#ef4444",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_df["year_month"],
            y=monthly_df["net_savings"],
            name="Net Savings",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
        )
    )
    fig.update_layout(
        title="Monthly Cashflow & Net Savings",
        barmode="group",
        xaxis_title="Month",
        yaxis_title="Amount",
        height=340,
        margin=dict(l=10, r=20, t=45, b=30),
    )
    return fig


def _build_forecast_chart(monthly_df: pd.DataFrame, forecast) -> go.Figure:
    fig = go.Figure()
    if not monthly_df.empty:
        fig.add_trace(
            go.Scatter(
                x=monthly_df["year_month"],
                y=monthly_df["expenses"],
                name="Actual Expenses",
                mode="lines+markers",
                line=dict(color="#f97316", width=2.5),
            )
        )

    if forecast.target_month not in {"N/A", ""}:
        fig.add_trace(
            go.Scatter(
                x=[forecast.target_month],
                y=[forecast.predicted_total_expense],
                name="Forecast",
                mode="markers",
                marker=dict(color="#a855f7", size=12, symbol="diamond"),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[max(0.0, forecast.upper_bound - forecast.predicted_total_expense)],
                    arrayminus=[max(0.0, forecast.predicted_total_expense - forecast.lower_bound)],
                    color="#a855f7",
                ),
            )
        )

    fig.update_layout(
        title=f"Expense Trajectory & Forecast ({forecast.target_month})",
        xaxis_title="Month",
        yaxis_title="Expenses",
        height=340,
        margin=dict(l=10, r=20, t=45, b=30),
    )
    return fig


def _build_budget_chart(breakdown_df: pd.DataFrame, month_label: str) -> go.Figure:
    active = breakdown_df[(breakdown_df["spent"] > 0) | (breakdown_df["monthly_budget"] > 0)].copy()
    active = active.iloc[::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=active["category"],
            x=active["monthly_budget"],
            name="Monthly Budget",
            orientation="h",
            marker_color="#64748b",
        )
    )
    fig.add_trace(
        go.Bar(
            y=active["category"],
            x=active["spent"],
            name="Actual Spent",
            orientation="h",
            marker_color="#3b82f6",
        )
    )
    fig.update_layout(
        title=f"Budget vs. Actual Spending ({month_label})",
        barmode="group",
        xaxis_title="Amount",
        yaxis_title="",
        height=360,
        margin=dict(l=10, r=20, t=45, b=30),
    )
    return fig


init_db(DEFAULT_DB_PATH)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Controls & AI Config")

    if st.button("Seed 12-Month Sample Data", use_container_width=True):
        added = seed_sample_data(DEFAULT_DB_PATH)
        export_sample_csv(SAMPLE_CSV_PATH)
        st.success(f"Seeded {added} new sample transactions.")
        st.rerun()

    st.divider()
    use_llm = st.toggle("Enable AI Advisor (Gemini)", value=True)
    api_override = st.text_input("Gemini API Key (optional)", value="", type="password")

    st.divider()
    st.markdown(
        "**Stack**\n"
        "- **DB:** SQLite + SQL aggregations\n"
        "- **ML Classifier:** TF-IDF + LogisticRegression\n"
        "- **Forecast:** Ridge time features + walk-forward\n"
        f"- **Advisor:** `{DEFAULT_LLM_MODEL}` + Pydantic"
    )

# ---------- LOAD DATA ----------
all_tx_df = get_transactions_df(DEFAULT_DB_PATH)
monthly_df = get_monthly_summary_df(DEFAULT_DB_PATH)
categories_df = get_categories_df(DEFAULT_DB_PATH)
category_names = categories_df["name"].tolist() if not categories_df.empty else ["Shopping & Misc"]
budget_map = (
    dict(zip(categories_df["name"], categories_df["monthly_budget"]))
    if not categories_df.empty
    else {}
)

st.title("AI Personal Finance Manager")
st.caption(
    "SQLite transaction storage, hybrid ML expense categorization, walk-forward expense "
    "forecasting, anomaly detection, and grounded AI financial advice."
)

if all_tx_df.empty:
    st.info(
        "Your database has no transactions yet. Click **Seed 12-Month Sample Data** in the "
        "sidebar or import a CSV in the **Transactions & ML** tab to get started."
    )
    st.stop()

available_months = sorted(all_tx_df["year_month"].unique().tolist(), reverse=True)
selected_month = st.selectbox("Focus Month", options=available_months, index=0)

month_row = monthly_df[monthly_df["year_month"] == selected_month]
inc_val = float(month_row.iloc[0]["income"]) if not month_row.empty else 0.0
exp_val = float(month_row.iloc[0]["expenses"]) if not month_row.empty else 0.0
net_val = float(month_row.iloc[0]["net_savings"]) if not month_row.empty else 0.0
sav_rate = float(month_row.iloc[0]["savings_rate"]) if not month_row.empty else 0.0

breakdown_df = get_category_breakdown_df(DEFAULT_DB_PATH, month=selected_month)
forecast = forecast_monthly_expenses(all_tx_df, category_budgets=budget_map)
anomalies = detect_expense_anomalies(all_tx_df)

# ---------- TOP KPI ROW ----------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(f"Income ({selected_month})", f"{inc_val:.2f}")
k2.metric(f"Expenses ({selected_month})", f"{exp_val:.2f}")
k3.metric("Net Savings", f"{net_val:.2f}", f"{sav_rate:.1f}% savings rate")
k4.metric(
    f"Forecast ({forecast.target_month})",
    f"{forecast.predicted_total_expense:.2f}",
    f"±{max(0.0, forecast.upper_bound - forecast.predicted_total_expense):.0f} band",
)
k5.metric("Flagged Anomalies", f"{len(anomalies)}", "across history")

tab_overview, tab_categories, tab_tx, tab_advisor = st.tabs(
    ["Overview & Forecast", "Categories & Anomalies", "Transactions & ML", "AI Financial Advisor"]
)

# ---------- TAB 1: OVERVIEW & FORECAST ----------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_build_cashflow_chart(monthly_df), use_container_width=True)
    with c2:
        st.plotly_chart(_build_forecast_chart(monthly_df, forecast), use_container_width=True)

    mae_str = f"{forecast.walk_forward_mae:.2f}" if forecast.walk_forward_mae is not None else "N/A"
    mape_str = f"{forecast.walk_forward_mape:.1f}%" if forecast.walk_forward_mape is not None else "N/A"
    st.caption(
        f"Forecast model: `{forecast.method}` across {forecast.historical_months} month(s) | "
        f"Walk-forward MAE: **{mae_str}** | MAPE: **{mape_str}**"
    )

    if forecast.by_category:
        st.markdown(f"#### Predicted Expenses by Category ({forecast.target_month})")
        fc_df = pd.DataFrame(
            [{"Category": k, "Predicted Expense": v} for k, v in forecast.by_category.items()]
        )
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

# ---------- TAB 2: CATEGORIES & ANOMALIES ----------
with tab_categories:
    st.plotly_chart(_build_budget_chart(breakdown_df, selected_month), use_container_width=True)

    b_left, b_right = st.columns([1.4, 1])
    with b_left:
        st.markdown(f"#### Budget Status ({selected_month})")
        for _, row in breakdown_df.iterrows():
            cat_name = str(row["category"])
            spent = float(row["spent"])
            budget = float(row["monthly_budget"])
            if budget > 0:
                ratio = spent / budget
                st.write(f"**{cat_name}**: {spent:.2f} / {budget:.2f} ({ratio * 100:.0f}%)")
                st.progress(min(1.0, max(0.0, ratio)))
            elif spent > 0:
                st.write(f"**{cat_name}**: {spent:.2f} (no budget set)")

    with b_right:
        st.markdown("#### Update Monthly Budget")
        expense_cats = categories_df[categories_df["kind"] == "expense"]["name"].tolist()
        if expense_cats:
            target_cat = st.selectbox("Expense Category", options=expense_cats, key="budget_cat")
            current_b = float(budget_map.get(target_cat, 100.0))
            new_b = st.number_input(
                "Monthly Budget",
                min_value=0.0,
                value=current_b,
                step=10.0,
                key="budget_val",
            )
            if st.button("Save Budget", use_container_width=True):
                update_category_budget(target_cat, new_b, DEFAULT_DB_PATH)
                st.success(f"Updated {target_cat} budget to {new_b:.2f}.")
                st.rerun()

    st.divider()
    st.markdown("#### Unusual Transactions / Spending Spikes")
    if not anomalies:
        st.success("No unusual expense spikes detected.")
    else:
        anom_rows = [
            {
                "Date": a.tx_date,
                "Merchant": a.merchant,
                "Category": a.category,
                "Amount": a.amount,
                "Category Median": a.category_median,
                "Severity": a.severity.upper(),
                "Reason": a.reason,
            }
            for a in anomalies
        ]
        st.dataframe(pd.DataFrame(anom_rows), use_container_width=True, hide_index=True)

# ---------- TAB 3: TRANSACTIONS & ML ----------
with tab_tx:
    top_l, top_r = st.columns(2)

    with top_l:
        st.markdown("#### Add Single Transaction")
        with st.form("add_tx_form", clear_on_submit=True):
            f_date = st.date_input("Date")
            f_merchant = st.text_input("Merchant / Concept", placeholder="e.g., MERCADONA SUPERMERCADO")
            f_amount = st.number_input("Amount", min_value=0.01, value=25.0, step=1.0)
            f_type = st.selectbox("Type", options=["expense", "income"])
            f_cat = st.selectbox("Category", options=category_names)
            f_desc = st.text_input("Notes (optional)", placeholder="Optional note")
            submitted = st.form_submit_button("Add Transaction", use_container_width=True)
            if submitted and f_merchant.strip():
                row_id = add_transaction(
                    tx_date=f_date.isoformat(),
                    merchant=f_merchant,
                    amount=f_amount,
                    tx_type=f_type,
                    category_name=f_cat,
                    description=f_desc,
                    category_source="manual",
                    db_path=DEFAULT_DB_PATH,
                )
                if row_id:
                    st.success("Transaction saved.")
                else:
                    st.info("Duplicate transaction skipped.")
                st.rerun()

    with top_r:
        st.markdown("#### Import Bank CSV")
        st.caption("Expected columns: `date`, `merchant`, `amount` (optional: `type`, `category`, `description`).")
        uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_csv is not None and st.button("Import CSV", use_container_width=True):
            ins, skp = import_transactions_csv(uploaded_csv.getvalue(), DEFAULT_DB_PATH)
            st.success(f"Imported {ins} transactions ({skp} duplicates skipped).")
            st.rerun()

        if Path(SAMPLE_CSV_PATH).exists():
            st.download_button(
                "Download Sample CSV",
                data=Path(SAMPLE_CSV_PATH).read_bytes(),
                file_name="sample_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### Auto-Categorize with ML + Rules")
    clf = ExpenseCategorizer()
    fitted = clf.fit_from_dataframe(all_tx_df)
    eval_metrics = evaluate_ml_categorizer(all_tx_df)

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("ML Model Status", "Trained" if fitted else "Rule Fallback")
    m_col2.metric(
        "Holdout Accuracy",
        f"{eval_metrics.accuracy * 100:.1f}%" if eval_metrics else "N/A",
    )
    m_col3.metric(
        "Holdout Macro-F1",
        f"{eval_metrics.macro_f1 * 100:.1f}%" if eval_metrics else "N/A",
    )

    if st.button("Run Auto-Categorization on All Transactions", type="primary"):
        updated_count = 0
        for _, row in all_tx_df.iterrows():
            pred = clf.predict_one(
                merchant=str(row["merchant"]),
                description=str(row["description"]),
                tx_type=str(row["tx_type"]),
            )
            if pred.category != str(row["category"]):
                update_transaction_category(
                    tx_id=int(row["id"]),
                    category_name=pred.category,
                    category_source=pred.source,
                    db_path=DEFAULT_DB_PATH,
                )
                updated_count += 1
        st.success(f"Auto-categorization complete. Updated {updated_count} transaction(s).")
        st.rerun()

    st.markdown("#### Transaction Ledger")
    filter_cat = st.selectbox("Filter by Category", options=["All"] + category_names)
    view_df = get_transactions_df(DEFAULT_DB_PATH, month=selected_month, category_name=filter_cat)
    st.dataframe(view_df, use_container_width=True, hide_index=True)

# ---------- TAB 4: AI FINANCIAL ADVISOR ----------
with tab_advisor:
    resolved_key = _get_api_key(api_override)
    if st.button("Generate Financial Advice", type="primary"):
        with st.spinner("Analyzing cashflow, budgets, forecast, and anomalies..."):
            advice = generate_finance_advice(
                monthly_df=monthly_df,
                breakdown_df=breakdown_df,
                forecast=forecast,
                anomalies=anomalies,
                selected_month=selected_month,
                api_key=resolved_key,
                use_llm=use_llm,
            )
        st.session_state["finance_advice"] = advice

    if "finance_advice" in st.session_state:
        advice = st.session_state["finance_advice"]
        if advice.source == "llm":
            st.info(f"Generated via **{advice.model_used}** with strict Pydantic JSON validation.")
        else:
            st.warning(f"Deterministic mode active. ({advice.fallback_reason})")

        st.markdown("#### Monthly Health Summary")
        st.write(advice.monthly_health_summary)

        st.markdown("#### Anomaly & Spending Observations")
        for item in advice.anomaly_insights:
            st.markdown(f"- {item}")

        st.markdown("#### Category Recommendations")
        for rec in advice.category_recommendations:
            with st.container(border=True):
                st.markdown(f"**{rec.category}** — *{rec.estimated_monthly_impact}*")
                st.write(rec.suggested_action)

        st.markdown("#### Next Month Focus")
        for idx, step in enumerate(advice.next_month_focus, start=1):
            st.markdown(f"{idx}. {step}")