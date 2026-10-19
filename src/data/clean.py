
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "interim" / "retail_combined.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

NON_MERCHANDISE_CODES = {
    "M",
    "DOT",
    "POST",
    "AMAZONFEE",
    "BANK CHARGES",
}


def clean_data():
    print("=" * 60)
    print("RETAIL INTELLIGENCE - DATA CLEANING")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE,
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

    original_rows = len(df)

    # Standardize text fields
    for column in ["Invoice", "StockCode", "Description", "Country"]:
        df[column] = df[column].str.strip()

    # Create transaction quality flags
    df["is_cancelled"] = df["Invoice"].str.upper().str.startswith(
        "C", na=False
    )
    df["is_return_or_adjustment"] = df["Quantity"] < 0
    df["invalid_price"] = df["Price"] <= 0
    df["missing_customer"] = df["Customer ID"].isna()
    df["is_exact_duplicate"] = df.duplicated(
        subset=[
            "Invoice", "StockCode", "Description",
            "Quantity", "InvoiceDate", "Price",
            "Customer ID", "Country"
        ],
        keep="first"
    )

    # Identify unusually large quantities for review
    df["extreme_quantity"] = df["Quantity"].abs() > 1000

    # Valid positive sales
    valid_sales = (
        ~df["is_cancelled"]
        & ~df["is_return_or_adjustment"]
        & ~df["invalid_price"]
        & df["InvoiceDate"].notna()
        & df["StockCode"].notna()
    )

    sales = df.loc[valid_sales].copy()
    sales["is_non_merchandise"] = (
        sales["StockCode"]
        .str.upper()
        .isin(NON_MERCHANDISE_CODES)
    )
    merchandise = sales.loc[
        ~sales["is_non_merchandise"]
    ].copy()

    # Feature engineering
    sales["Revenue"] = sales["Quantity"] * sales["Price"]
    sales["Date"] = sales["InvoiceDate"].dt.date
    sales["Year"] = sales["InvoiceDate"].dt.year
    sales["Month"] = sales["InvoiceDate"].dt.month
    sales["DayOfWeek"] = sales["InvoiceDate"].dt.dayofweek

    merchandise["Revenue"] = (
        merchandise["Quantity"] * merchandise["Price"]
    )
    merchandise["Date"] = merchandise["InvoiceDate"].dt.date
    merchandise["Year"] = merchandise["InvoiceDate"].dt.year
    merchandise["Month"] = merchandise["InvoiceDate"].dt.month
    merchandise["DayOfWeek"] = merchandise["InvoiceDate"].dt.dayofweek

    # Customer segmentation dataset
    customers = sales.loc[
        ~sales["missing_customer"]
    ].copy()

    # Output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets
    sales.to_csv(
        OUTPUT_DIR / "retail_sales.csv",
        index=False
    )
    customers.to_csv(
        OUTPUT_DIR / "retail_customers.csv",
        index=False
    )
    merchandise.to_csv(
        OUTPUT_DIR / "retail_merchandise.csv",
        index=False
    )

    # Quality summary
    summary = {
        "original_rows": original_rows,
        "valid_sales_rows": len(sales),
        "merchandise_rows": len(merchandise),
        "non_merchandise_rows": int(
            sales["is_non_merchandise"].sum()
        ),
        "merchandise_revenue": float(
            merchandise["Revenue"].sum()
        ),
        "customer_rows": len(customers),
        "excluded_from_sales": original_rows - len(sales),
        "cancelled_rows": int(df["is_cancelled"].sum()),
        "negative_quantity_rows": int(
            df["is_return_or_adjustment"].sum()
        ),
        "non_positive_price_rows": int(
            df["invalid_price"].sum()
        ),
        "duplicate_rows_flagged": int(
            df["is_exact_duplicate"].sum()
        ),
        "extreme_quantity_rows": int(
            df["extreme_quantity"].sum()
        ),
        "gross_positive_sales": float(sales["Revenue"].sum()),
    }

    pd.DataFrame([summary]).to_csv(
        REPORT_DIR / "cleaning_summary.csv",
        index=False
    )

    print("\nCLEANING RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value:,}")

    print("\nOutput files:")
    print(OUTPUT_DIR / "retail_sales.csv")
    print(OUTPUT_DIR / "retail_customers.csv")
    print(OUTPUT_DIR / "retail_merchandise.csv")

    return sales, customers


if __name__ == "__main__":
    clean_data()
