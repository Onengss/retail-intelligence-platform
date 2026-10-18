
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "interim" / "retail_combined.csv"
REPORT_DIR = BASE_DIR / "reports"


def explore_data():
    print("=" * 60)
    print("RETAIL INTELLIGENCE - DATA QUALITY REPORT")
    print("=" * 60)

    df = pd.read_csv(
        DATA_FILE,
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Description": "string",
            "Customer ID": "string",
            "Country": "string",
        },
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset overview
    print("\n1. DATASET OVERVIEW")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    # 2. Missing values
    print("\n2. MISSING VALUES")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2),
    })
    print(missing)

    # 3. Duplicate records
    duplicate_count = df.duplicated().sum()
    print("\n3. DUPLICATE RECORDS")
    print(f"Exact duplicate rows: {duplicate_count:,}")

    # 4. Quantity and price anomalies
    print("\n4. QUANTITY AND PRICE CHECKS")
    print(f"Negative quantities: {(df['Quantity'] < 0).sum():,}")
    print(f"Zero quantities: {(df['Quantity'] == 0).sum():,}")
    print(f"Negative prices: {(df['Price'] < 0).sum():,}")
    print(f"Zero prices: {(df['Price'] == 0).sum():,}")

    # 5. Cancelled invoices
    cancelled = df["Invoice"].str.startswith(
        "C", na=False
    )

    print("\n5. CANCELLED INVOICES")
    print(f"Cancellation rows: {cancelled.sum():,}")
    print(
        f"Unique cancelled invoices: {df.loc[cancelled, 'Invoice'].nunique():,}")

    # 6. Business overview
    print("\n6. BUSINESS OVERVIEW")
    print(f"Unique invoices: {df['Invoice'].nunique():,}")
    print(f"Unique products: {df['StockCode'].nunique():,}")
    print(f"Unique customers: {df['Customer ID'].nunique():,}")
    print(f"Countries: {df['Country'].nunique():,}")

    print("\nTOP 10 COUNTRIES")
    print(df["Country"].value_counts().head(10))

    # 7. Numerical statistics
    print("\n7. NUMERICAL STATISTICS")
    print(df[["Quantity", "Price"]].describe())

    # Save reports
    missing.to_csv(REPORT_DIR / "missing_values.csv")

    summary = pd.DataFrame([{
        "total_rows": len(df),
        "duplicate_rows": int(duplicate_count),
        "negative_quantity_rows": int((df["Quantity"] < 0).sum()),
        "zero_price_rows": int((df["Price"] == 0).sum()),
        "negative_price_rows": int((df["Price"] < 0).sum()),
        "cancelled_rows": int(cancelled.sum()),
        "unique_invoices": df["Invoice"].nunique(),
        "unique_customers": df["Customer ID"].nunique(),
        "unique_products": df["StockCode"].nunique(),
    }])

    summary.to_csv(
        REPORT_DIR / "data_quality_summary.csv",
        index=False
    )

    print("\nReports saved successfully!")


if __name__ == "__main__":
    explore_data()
