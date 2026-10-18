
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data/processed/retail_merchandise.csv"

load_dotenv(BASE_DIR / ".env")


def load_to_postgres():
    print("Loading merchandise data into PostgreSQL...")

    columns = {
        "Invoice": "invoice",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "price",
        "Customer ID": "customer_id",
        "Country": "country",
        "Revenue": "revenue",
    }

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Customer ID": "string",
        },
        low_memory=False,
    )

    df = df[list(columns.keys())].rename(columns=columns)

    print(f"Rows to load: {len(df):,}")

    connection = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "retail_intelligence"),
        user=os.getenv("POSTGRES_USER", "retail_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    try:
        with connection:
            with connection.cursor() as cursor:
                # Replace data on each run to avoid duplicate loads
                cursor.execute(
                    "TRUNCATE TABLE retail_transactions RESTART IDENTITY")

                # Bulk copy in manageable chunks
                from io import StringIO

                for start in range(0, len(df), 100000):
                    chunk = df.iloc[start:start + 100000]

                    buffer = StringIO()
                    chunk.to_csv(
                        buffer,
                        index=False,
                        header=False,
                        na_rep=""
                    )
                    buffer.seek(0)

                    cursor.copy_expert(
                        """
                        COPY retail_transactions (
                            invoice, stock_code, description,
                            quantity, invoice_date, price,
                            customer_id, country, revenue
                        )
                        FROM STDIN WITH (FORMAT CSV, NULL '')
                        """,
                        buffer,
                    )

                    print(
                        f"Loaded {min(start + 100000, len(df)):,} rows"
                    )

                cursor.execute(
                    "SELECT COUNT(*), SUM(revenue) FROM retail_transactions"
                )
                count, revenue = cursor.fetchone()

                print(f"\nDatabase rows: {count:,}")
                print(f"Database revenue: £{revenue:,.2f}")

                if count != len(df):
                    raise ValueError(
                        f"Row mismatch: expected {len(df)}, got {count}"
                    )

        print("\nPostgreSQL load completed successfully!")

    finally:
        connection.close()


if __name__ == "__main__":
    load_to_postgres()
