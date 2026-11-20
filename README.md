# AI Personal Finance Manager

An end-to-end **data & AI personal finance application** combining a relational **SQLite** analytics layer, a hybrid **TF-IDF + LogisticRegression** expense categorizer, **walk-forward** monthly expense forecasting with anomaly detection, and **Gemini** savings recommendations grounded via **Pydantic** JSON schemas.

## Architecture

```text
CSV / Manual Input ──► [1. SQLite Layer] ──► Normalized Tables + SQL Aggregations
                                                      │
                                                      ▼
                       [2. Hybrid Categorizer] ──► Rules + TF-IDF Word/Char + LogisticRegression
                                                      │
                                                      ▼
                       [3. Forecaster & Anomalies] ──► Ridge Time Features + Walk-Forward MAE/MAPE + Outliers
                                                      │
                                                      ▼
                       [4. AI Financial Advisor] ──► Gemini + Pydantic JSON (with deterministic fallback)
                                                      │
                                                      ▼
                       [5. Streamlit Dashboard] ──► Interactive Plotly Charts, Budgets & Ledger
```

## Key Features

- **SQLite Analytics & Deduplication (`src/database.py`):** Manages `categories` and `transactions` with deterministic deduplication hashes, CSV bank statement import, budget tracking, and monthly SQL aggregations (`income`, `expenses`, `net_savings`, `savings_rate`).
- **Hybrid Expense Categorizer (`src/categorizer.py`):** Combines merchant keyword rules with a `scikit-learn` pipeline (`FeatureUnion` of word and character n-gram `TF-IDF` + balanced `LogisticRegression`), holdout evaluation (`Accuracy` and `Macro-F1`), and confidence scores.
- **Expense Forecasting & Anomaly Detection (`src/forecaster.py`):** Forecasts next-month total and per-category expenses using Ridge regression with trend and cyclical month features, evaluated with **strict walk-forward validation** (`MAE` and `MAPE`) without future data leakage, alongside robust outlier detection (`MAD` / median multiplier).
- **Grounded AI Financial Advisor (`src/advisor.py`):** Uses `google-genai` with a strict `Pydantic` schema (`FinanceAdvisorPayload`) grounded in SQL summaries, budget usage, forecast intervals, and flagged anomalies—with a deterministic rule-based fallback when no API key is set.
- **Interactive Dashboard (`app.py`):** Built with **Streamlit** and **Plotly** to explore monthly cashflow, forecast uncertainty bands, budget progress bars, CSV imports, ML auto-categorization, and AI recommendations.

## Screenshots

![Dashboard & KPIs](docs/overview_kpis.png)
![Cashflow & Forecast Charts](docs/charts_forecast.png)
![Predicted Expenses by Category](docs/category_forecast.png)
![Budget vs. Actual Spending](docs/budgets_categories.png)
![Add Transaction & Import CSV](docs/transactions_import.png)

## Quickstart

### Local Setup

```bash
git clone https://github.com/paumerino/ai-finance-manager.git
cd ai-finance-manager
python -m venv .venv

# Windows (CMD)
.\.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

*(Optional)* Enable Gemini-powered advice by creating a `.env` file from `.env.example`:

```env
GEMINI_API_KEY=your_api_key_here
```

Seed sample data, run the test suite, and start the dashboard:

```bash
python -m src.sample_data
python -m pytest -v
streamlit run app.py
```

### Run with Docker

```bash
docker build -t ai-finance-manager .
docker run -p 8501:8501 --env-file .env ai-finance-manager
```

## Tech Stack

`Python` · `SQLite` · `pandas` · `NumPy` · `scikit-learn` · `Pydantic` · `Google GenAI` · `Streamlit` · `Plotly` · `pytest` · `Docker`