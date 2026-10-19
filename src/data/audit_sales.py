
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data/processed/retail_sales.csv"
REPORT_DIR = BASE_DIR / "reports"


def audit_sales():
    df = pd.read_csv(
        DATA_FILE,
        dtype={"StockCode": "string", "Invoice": "string"},
        low_memory=False
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTOP 15 TRANSACTIONS BY REVENUE")
    columns = [
        "Invoice", "StockCode", "Description",
        "Quantity", "Price", "Revenue"
    ]

    top_transactions = df.nlargest(15, "Revenue")[columns]
    print(top_transactions.to_string(index=False))

    print("\nTOP 20 STOCK CODES BY REVENUE")
    top_products = (
        df.groupby("StockCode", dropna=False)
        .agg(
            total_revenue=("Revenue", "sum"),
            total_quantity=("Quantity", "sum"),
            transaction_rows=("Invoice", "size")
        )
        .sort_values("total_revenue", ascending=False)
        .head(20)
    )

    print(top_products.to_string())

    print("\nTOP 20 DESCRIPTIONS BY REVENUE")
    top_descriptions = (
        df.groupby("Description", dropna=False)["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print(top_descriptions.to_string())

    print("\nEXTREME QUANTITY TRANSACTIONS")
    extreme = df[df["Quantity"].abs() > 1000]
    total_revenue = df["Revenue"].sum()
    extreme_revenue = extreme["Revenue"].sum()
    print(f"Extreme quantity rows: {len(extreme):,}")
    print(f"Extreme quantity revenue: {extreme_revenue:,.2f}")
    if total_revenue:
        print(
            "Share of total revenue: "
            f"{extreme_revenue / total_revenue:.2%}"
        )
    else:
        print("Share of total revenue: unavailable (total revenue is zero)")
    print(
        extreme.nlargest(15, "Revenue")[columns]
        .to_string(index=False)
    )

    top_transactions.to_csv(
        REPORT_DIR / "top_transactions_audit.csv",
        index=False
    )

    top_products.to_csv(
        REPORT_DIR / "top_stock_codes_audit.csv"
    )

    print("\nAudit reports saved successfully.")


if __name__ == "__main__":
    audit_sales()
