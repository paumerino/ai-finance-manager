"""Realistic synthetic transaction generator and sample CSV exporter."""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from src.database import DEFAULT_DB_PATH, add_transaction, init_db

SAMPLE_CSV_PATH = Path("data/sample_transactions.csv")

MERCHANT_TEMPLATES: dict[str, list[tuple[str, float, float, str]]] = {
    "Groceries": [
        ("MERCADONA SUPERMERCADO", 38.0, 72.0, "expense"),
        ("LIDL EXPRESS", 22.0, 52.0, "expense"),
        ("CARREFOUR MARKET", 30.0, 65.0, "expense"),
        ("BIEDRONKA MARKET", 18.0, 42.0, "expense"),
    ],
    "Transport": [
        ("RENFE CERCANIAS", 12.0, 28.0, "expense"),
        ("METRO BARCELONA T-USUAL", 21.5, 22.0, "expense"),
        ("UBER TRIP", 9.0, 24.0, "expense"),
        ("ZTM POZNAN TICKET", 10.0, 20.0, "expense"),
    ],
    "Dining & Leisure": [
        ("RESTAURANTE EL PUERTO", 16.0, 34.0, "expense"),
        ("CAFE CENTRAL", 4.5, 11.0, "expense"),
        ("CINESA LUXE", 11.0, 22.0, "expense"),
        ("PIZZERIA NAPOLI", 14.0, 28.0, "expense"),
    ],
    "Subscriptions & Tech": [
        ("SPOTIFY AB", 10.99, 10.99, "expense"),
        ("GITHUB PRO", 4.0, 4.0, "expense"),
        ("CLOUD STORAGE 200GB", 2.99, 2.99, "expense"),
    ],
    "Health & Fitness": [
        ("GYM MEMBERSHIP", 29.9, 29.9, "expense"),
        ("FARMACIA CENTRAL", 8.0, 26.0, "expense"),
    ],
    "Shopping & Misc": [
        ("AMAZON EU", 19.0, 68.0, "expense"),
        ("DECATHLON SPORTS", 20.0, 55.0, "expense"),
        ("LIBRERIA TECNICA", 15.0, 40.0, "expense"),
    ],
}


def _month_starts(months_back: int = 12, ref_date: date | None = None) -> list[date]:
    """Return the first day of each month covering the last `months_back` months."""
    today = ref_date or date.today()
    year, month = today.year, today.month
    starts: list[date] = []
    for _ in range(months_back):
        starts.append(date(year, month, 1))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    starts.reverse()
    return starts


def seed_sample_data(
    db_path: str | Path = DEFAULT_DB_PATH,
    months: int = 12,
    seed: int = 42,
) -> int:
    """Populate SQLite with realistic monthly income, rent, and variable expenses."""
    init_db(db_path)
    rng = random.Random(seed)
    inserted = 0

    for m_start in _month_starts(months_back=months):
        days_in_month = (
            (date(m_start.year + (m_start.month // 12), (m_start.month % 12) + 1, 1) - m_start).days
        )

        # Monthly stipend / part-time income on day 1
        income_amount = round(rng.uniform(1050.0, 1300.0), 2)
        if add_transaction(
            tx_date=m_start.isoformat(),
            merchant="UNIVERSITY STIPEND / PAYROLL",
            amount=income_amount,
            tx_type="income",
            category_name="Income",
            description="Monthly income",
            category_source="rule",
            db_path=db_path,
        ):
            inserted += 1

        # Fixed housing & utilities on day 3
        rent_amount = round(rng.uniform(540.0, 610.0), 2)
        rent_day = m_start + timedelta(days=min(2, days_in_month - 1))
        if add_transaction(
            tx_date=rent_day.isoformat(),
            merchant="ALQUILER APARTAMENTO / RENT",
            amount=rent_amount,
            tx_type="expense",
            category_name="Housing & Utilities",
            description="Monthly rent and utilities",
            category_source="rule",
            db_path=db_path,
        ):
            inserted += 1

        # Variable expenses across the month
        for category, items in MERCHANT_TEMPLATES.items():
            tx_count_for_cat = rng.randint(2, 5)
            for _ in range(tx_count_for_cat):
                merchant, low, high, tx_type = rng.choice(items)
                amount = round(rng.uniform(low, high), 2)
                day_offset = rng.randint(0, max(0, days_in_month - 1))
                tx_day = m_start + timedelta(days=day_offset)
                if add_transaction(
                    tx_date=tx_day.isoformat(),
                    merchant=merchant,
                    amount=amount,
                    tx_type=tx_type,
                    category_name=category,
                    description="Synthetic sample transaction",
                    category_source="rule",
                    db_path=db_path,
                ):
                    inserted += 1

        # Inject an occasional outlier expense every 4th month for anomaly testing
        if m_start.month % 4 == 0:
            outlier_day = m_start + timedelta(days=min(14, days_in_month - 1))
            if add_transaction(
                tx_date=outlier_day.isoformat(),
                merchant="MEDIA MARKT LAPTOP REPAIR",
                amount=340.0,
                tx_type="expense",
                category_name="Shopping & Misc",
                description="Unusual hardware expense",
                category_source="rule",
                db_path=db_path,
            ):
                inserted += 1

    return inserted


def export_sample_csv(csv_path: str | Path = SAMPLE_CSV_PATH) -> Path:
    """Create a sample CSV file that users can upload via the dashboard."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["date", "merchant", "amount", "type", "category", "description"],
        ["2026-10-01", "UNIVERSITY GRANT", "1150.00", "income", "Income", "Monthly stipend"],
        ["2026-10-03", "ALQUILER PISO", "575.00", "expense", "Housing & Utilities", "Rent"],
        ["2026-10-06", "MERCADONA SUPERMERCADO", "48.20", "expense", "Groceries", "Weekly groceries"],
        ["2026-10-09", "RENFE CERCANIAS", "14.50", "expense", "Transport", "Train ticket"],
        ["2026-10-12", "RESTAURANTE EL PUERTO", "24.00", "expense", "Dining & Leisure", "Dinner"],
        ["2026-10-15", "SPOTIFY AB", "10.99", "expense", "Subscriptions & Tech", "Music plan"],
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    return out


if __name__ == "__main__":
    count = seed_sample_data()
    csv_file = export_sample_csv()
    print(f"Inserted {count} sample transactions into {DEFAULT_DB_PATH} and wrote {csv_file}.")