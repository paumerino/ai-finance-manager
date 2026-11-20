"""Unit tests for SQLite database schema, CSV import, deduplication, and SQL summaries."""

from pathlib import Path

import pytest

from src.database import (
    add_transaction,
    get_category_breakdown_df,
    get_monthly_summary_df,
    get_transactions_df,
    import_transactions_csv,
    init_db,
    update_category_budget,
)


def test_init_db_seeds_default_categories(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    init_db(db_file)
    init_db(db_file)  # idempotent second call

    breakdown = get_category_breakdown_df(db_file)
    assert not breakdown.empty
    assert "Groceries" in set(breakdown["category"])


def test_add_transaction_deduplicates_identical_entries(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    first_id = add_transaction("2026-10-05", "Mercadona", 45.5, "expense", "Groceries", db_path=db_file)
    dup_id = add_transaction("2026-10-05", "MERCADONA", 45.5, "expense", "Groceries", db_path=db_file)

    assert first_id is not None
    assert dup_id is None

    df = get_transactions_df(db_file)
    assert len(df) == 1
    assert df.iloc[0]["tx_date"] == "2026-10-05"
    assert df.iloc[0]["amount"] == 45.5


def test_csv_import_and_monthly_sql_summary(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    csv_content = (
        "date,merchant,amount,type,category\n"
        "2026-09-01,Payroll,1200.00,income,Income\n"
        "2026-09-04,Mercadona,80.00,expense,Groceries\n"
        "2026-09-10,Metro Ticket,20.00,expense,Transport\n"
        "2026-09-10,Metro Ticket,20.00,expense,Transport\n"
    )
    inserted, skipped = import_transactions_csv(csv_content, db_path=db_file)
    assert inserted == 3
    assert skipped == 1

    monthly = get_monthly_summary_df(db_file)
    assert len(monthly) == 1
    row = monthly.iloc[0]
    assert row["year_month"] == "2026-09"
    assert row["income"] == 1200.0
    assert row["expenses"] == 100.0
    assert row["net_savings"] == 1100.0
    assert row["savings_rate"] == pytest.approx(91.7, abs=0.1)


def test_budget_update_and_category_breakdown(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    update_category_budget("Groceries", 200.0, db_path=db_file)
    add_transaction("2026-10-02", "Lidl", 150.0, "expense", "Groceries", db_path=db_file)

    breakdown = get_category_breakdown_df(db_file, month="2026-10")
    groceries_row = breakdown[breakdown["category"] == "Groceries"].iloc[0]
    assert groceries_row["monthly_budget"] == 200.0
    assert groceries_row["spent"] == 150.0
    assert groceries_row["remaining"] == 50.0
    assert groceries_row["budget_usage_pct"] == 75.0