"""SQLite database layer, CSV ingestion, and analytical queries for personal finance."""

from __future__ import annotations

import csv
import io
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

DEFAULT_DB_PATH = Path("data/finance.db")

DEFAULT_CATEGORIES: list[tuple[str, str, float]] = [
    ("Housing & Utilities", "expense", 650.0),
    ("Groceries", "expense", 250.0),
    ("Transport", "expense", 100.0),
    ("Dining & Leisure", "expense", 150.0),
    ("Subscriptions & Tech", "expense", 60.0),
    ("Health & Fitness", "expense", 60.0),
    ("Shopping & Misc", "expense", 120.0),
    ("Income", "income", 0.0),
]

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('income', 'expense')),
    monthly_budget REAL NOT NULL DEFAULT 0.0 CHECK (monthly_budget >= 0.0)
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_date TEXT NOT NULL,
    merchant TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    amount REAL NOT NULL CHECK (amount > 0.0),
    tx_type TEXT NOT NULL CHECK (tx_type IN ('income', 'expense')),
    category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    category_source TEXT NOT NULL DEFAULT 'manual' CHECK (category_source IN ('manual', 'ml', 'rule')),
    tx_hash TEXT UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(tx_date);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(tx_type);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category_id);
"""


@contextmanager
def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with foreign keys and row access enabled."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create tables/indexes and seed default categories idempotently."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        for name, kind, budget in DEFAULT_CATEGORIES:
            conn.execute(
                """
                INSERT INTO categories (name, kind, monthly_budget)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (name, kind, budget),
            )


def _normalize_date(raw_date: str) -> str:
    """Parse common date formats into ISO YYYY-MM-DD."""
    parsed = pd.to_datetime(str(raw_date).strip(), format="mixed", errors="raise")
    return parsed.strftime("%Y-%m-%d")


def _build_tx_hash(tx_date: str, merchant: str, amount: float, tx_type: str) -> str:
    """Deterministic fingerprint to avoid duplicate imports of the same bank line."""
    clean_merchant = " ".join(merchant.strip().upper().split())
    return f"{tx_date}|{tx_type}|{amount:.2f}|{clean_merchant}"


def get_categories_df(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return all categories and their monthly budgets."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT id, name, kind, monthly_budget FROM categories ORDER BY kind, name",
            conn,
        )


def update_category_budget(
    category_name: str,
    monthly_budget: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Update the monthly budget target for a category."""
    if monthly_budget < 0:
        raise ValueError("Monthly budget cannot be negative.")
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE categories SET monthly_budget = ? WHERE name = ?",
            (float(monthly_budget), category_name),
        )
        if cur.rowcount == 0:
            raise KeyError(f"Category not found: {category_name}")


def add_transaction(
    tx_date: str,
    merchant: str,
    amount: float,
    tx_type: str = "expense",
    category_name: str | None = None,
    description: str = "",
    category_source: str = "manual",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int | None:
    """Insert a single transaction; returns row id or None if duplicate."""
    if amount <= 0:
        raise ValueError("Transaction amount must be positive.")
    if tx_type not in {"income", "expense"}:
        raise ValueError("tx_type must be 'income' or 'expense'.")
    if not merchant or not merchant.strip():
        raise ValueError("Merchant name cannot be empty.")

    init_db(db_path)
    iso_date = _normalize_date(tx_date)
    clean_merchant = " ".join(merchant.strip().split())
    tx_hash = _build_tx_hash(iso_date, clean_merchant, float(amount), tx_type)

    with get_connection(db_path) as conn:
        category_id: int | None = None
        if category_name:
            row = conn.execute(
                "SELECT id FROM categories WHERE name = ?",
                (category_name,),
            ).fetchone()
            if row:
                category_id = int(row["id"])

        cur = conn.execute(
            """
            INSERT INTO transactions (
                tx_date, merchant, description, amount, tx_type,
                category_id, category_source, tx_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tx_hash) DO NOTHING
            """,
            (
                iso_date,
                clean_merchant,
                description.strip(),
                round(float(amount), 2),
                tx_type,
                category_id,
                category_source,
                tx_hash,
            ),
        )
        return int(cur.lastrowid) if cur.rowcount and cur.lastrowid else None


def import_transactions_csv(
    csv_source: str | Path | bytes,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[int, int]:
    """Import transactions from a CSV file path, raw CSV string, or bytes.

    Expected columns: date, merchant, amount (optional: type, category, description).
    If 'amount' is negative and 'type' is omitted, treats it as an expense.
    Returns (inserted_count, skipped_duplicates_count).
    """
    init_db(db_path)
    if isinstance(csv_source, bytes):
        text = csv_source.decode("utf-8-sig")
    elif isinstance(csv_source, Path):
        text = csv_source.read_text(encoding="utf-8-sig")
    else:
        p = Path(csv_source)
        text = p.read_text(encoding="utf-8-sig") if p.exists() else csv_source

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV file is empty or missing headers.")

    normalized_headers = {h.strip().lower(): h for h in reader.fieldnames}
    for required in ("date", "merchant", "amount"):
        if required not in normalized_headers:
            raise ValueError(f"CSV missing required column: '{required}'.")

    inserted = 0
    skipped = 0

    for row in reader:
        raw_date = (row.get(normalized_headers["date"]) or "").strip()
        raw_merchant = (row.get(normalized_headers["merchant"]) or "").strip()
        raw_amount_str = (row.get(normalized_headers["amount"]) or "").strip().replace(",", ".")
        if not raw_date or not raw_merchant or not raw_amount_str:
            continue

        numeric_amount = float(raw_amount_str)
        explicit_type = ""
        if "type" in normalized_headers:
            explicit_type = (row.get(normalized_headers["type"]) or "").strip().lower()

        if explicit_type in {"income", "expense"}:
            tx_type = explicit_type
            abs_amount = abs(numeric_amount)
        else:
            tx_type = "expense" if numeric_amount < 0 else "income"
            abs_amount = abs(numeric_amount)

        if abs_amount == 0:
            continue

        category_name = None
        if "category" in normalized_headers:
            cat_val = (row.get(normalized_headers["category"]) or "").strip()
            if cat_val:
                category_name = cat_val

        description = ""
        if "description" in normalized_headers:
            description = (row.get(normalized_headers["description"]) or "").strip()

        row_id = add_transaction(
            tx_date=raw_date,
            merchant=raw_merchant,
            amount=abs_amount,
            tx_type=tx_type,
            category_name=category_name,
            description=description,
            category_source="manual" if category_name else "rule",
            db_path=db_path,
        )
        if row_id is not None:
            inserted += 1
        else:
            skipped += 1

    return inserted, skipped


def update_transaction_category(
    tx_id: int,
    category_name: str,
    category_source: str = "manual",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Assign or update the category of an existing transaction."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cat_row = conn.execute(
            "SELECT id FROM categories WHERE name = ?",
            (category_name,),
        ).fetchone()
        if not cat_row:
            raise KeyError(f"Unknown category: {category_name}")

        conn.execute(
            "UPDATE transactions SET category_id = ?, category_source = ? WHERE id = ?",
            (int(cat_row["id"]), category_source, int(tx_id)),
        )


def get_transactions_df(
    db_path: str | Path = DEFAULT_DB_PATH,
    month: str | None = None,
    category_name: str | None = None,
) -> pd.DataFrame:
    """Fetch transactions joined with category metadata, optionally filtered."""
    init_db(db_path)
    query = """
        SELECT
            t.id,
            t.tx_date,
            substr(t.tx_date, 1, 7) AS year_month,
            t.merchant,
            t.description,
            t.amount,
            t.tx_type,
            COALESCE(c.name, 'Uncategorized') AS category,
            t.category_source
        FROM transactions t
        LEFT JOIN categories c ON t.category_id = c.id
        WHERE 1=1
    """
    params: list[object] = []
    if month:
        query += " AND substr(t.tx_date, 1, 7) = ?"
        params.append(month)
    if category_name and category_name != "All":
        query += " AND COALESCE(c.name, 'Uncategorized') = ?"
        params.append(category_name)

    query += " ORDER BY t.tx_date DESC, t.id DESC"

    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_monthly_summary_df(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Aggregate monthly income, expenses, net savings, and savings rate using SQL."""
    init_db(db_path)
    query = """
        SELECT
            substr(tx_date, 1, 7) AS year_month,
            ROUND(SUM(CASE WHEN tx_type = 'income' THEN amount ELSE 0.0 END), 2) AS income,
            ROUND(SUM(CASE WHEN tx_type = 'expense' THEN amount ELSE 0.0 END), 2) AS expenses,
            ROUND(
                SUM(CASE WHEN tx_type = 'income' THEN amount ELSE 0.0 END)
                - SUM(CASE WHEN tx_type = 'expense' THEN amount ELSE 0.0 END),
                2
            ) AS net_savings,
            COUNT(*) AS tx_count
        FROM transactions
        GROUP BY substr(tx_date, 1, 7)
        ORDER BY year_month ASC
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return pd.DataFrame(
            columns=["year_month", "income", "expenses", "net_savings", "savings_rate", "tx_count"]
        )

    df["savings_rate"] = df.apply(
        lambda r: round((r["net_savings"] / r["income"]) * 100.0, 1) if r["income"] > 0 else 0.0,
        axis=1,
    )
    return df


def get_category_breakdown_df(
    db_path: str | Path = DEFAULT_DB_PATH,
    month: str | None = None,
) -> pd.DataFrame:
    """Compare actual spending per expense category against monthly budgets."""
    init_db(db_path)
    date_filter = " AND substr(t.tx_date, 1, 7) = ?" if month else ""
    params: list[object] = [month] if month else []

    query = f"""
        SELECT
            c.name AS category,
            c.monthly_budget,
            ROUND(COALESCE(SUM(CASE WHEN t.tx_type = 'expense' THEN t.amount ELSE 0.0 END), 0.0), 2) AS spent,
            COUNT(t.id) AS tx_count
        FROM categories c
        LEFT JOIN transactions t
            ON t.category_id = c.id {date_filter}
        WHERE c.kind = 'expense'
        GROUP BY c.id, c.name, c.monthly_budget
        ORDER BY spent DESC, c.name ASC
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["remaining"] = (df["monthly_budget"] - df["spent"]).round(2)
    df["budget_usage_pct"] = df.apply(
        lambda r: round((r["spent"] / r["monthly_budget"]) * 100.0, 1)
        if r["monthly_budget"] > 0
        else 0.0,
        axis=1,
    )
    return df