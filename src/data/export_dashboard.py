
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "dashboard" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "retail_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "retail_intelligence"),
    )
    return create_engine(url)


def main():
    queries = {
        "overview_kpi": """
            SELECT
                COUNT(DISTINCT invoice) AS orders,
                COUNT(DISTINCT customer_id) AS customers,
                ROUND(SUM(revenue), 2) AS revenue,
                SUM(quantity) AS units
            FROM retail_transactions
        """,
        "overview_monthly": """
            SELECT
                DATE_TRUNC('month', invoice_date)::date AS month,
                SUM(revenue)::float AS revenue
            FROM retail_transactions
            GROUP BY 1
            ORDER BY 1
        """,
        "overview_countries": """
            SELECT
                country,
                SUM(revenue)::float AS revenue
            FROM retail_transactions
            GROUP BY country
            ORDER BY revenue DESC
            LIMIT 10
        """,
        "customer_segments": """
            SELECT
                segment,
                COUNT(*) AS customers,
                SUM(monetary)::float AS revenue,
                AVG(recency)::float AS avg_recency,
                AVG(frequency)::float AS avg_frequency,
                AVG(monetary)::float AS avg_spend
            FROM customer_rfm
            GROUP BY segment
            ORDER BY revenue DESC
        """,
    }

    engine = get_engine()

    with engine.connect() as conn:
        for name, query in queries.items():
            df = pd.read_sql_query(text(query), conn)
            path = OUTPUT_DIR / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"Exported {path.name}: {len(df)} rows")

    engine.dispose()
    print("Dashboard data export completed.")


if __name__ == "__main__":
    main()
