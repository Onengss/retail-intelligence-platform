
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data/processed/retail_merchandise.csv"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def run_business_eda():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Description": "string",
            "Country": "string",
        },
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )

    print("=" * 60)
    print("RETAIL INTELLIGENCE - BUSINESS EDA")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Revenue: £{df['Revenue'].sum():,.2f}")

    # 1. Monthly revenue trend
    monthly = (
        df.set_index("InvoiceDate")
        .resample("MS")["Revenue"]
        .sum()
        .rename("Revenue")
        .reset_index()
    )

    monthly.to_csv(
        REPORT_DIR / "monthly_revenue.csv",
        index=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        monthly["InvoiceDate"],
        monthly["Revenue"],
        marker="o"
    )
    plt.title("Monthly Merchandise Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "monthly_revenue.png",
        dpi=160
    )
    plt.close()

    # 2. Top 10 products
    top_products = (
        df.groupby("Description", dropna=False)["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    top_products.to_csv(
        REPORT_DIR / "top_products.csv"
    )

    plt.figure(figsize=(12, 6))
    top_products.plot(kind="barh")
    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Revenue (£)")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "top_products.png",
        dpi=160
    )
    plt.close()

    # 3. Country performance
    country_sales = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    country_sales.to_csv(
        REPORT_DIR / "country_sales.csv"
    )

    top_countries = country_sales.head(10).sort_values()

    plt.figure(figsize=(10, 6))
    top_countries.plot(kind="barh")
    plt.title("Top 10 Countries by Merchandise Revenue")
    plt.xlabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "country_sales.png",
        dpi=160
    )
    plt.close()

    # 4. Day-of-week sales
    daily_sales = (
        df.groupby("DayOfWeek")["Revenue"]
        .sum()
        .reindex(range(7), fill_value=0)
    )

    daily_sales.index = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    daily_sales.to_csv(
        REPORT_DIR / "weekday_sales.csv"
    )

    plt.figure(figsize=(10, 5))
    daily_sales.plot(kind="bar")
    plt.title("Total Merchandise Revenue by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "weekday_sales.png",
        dpi=160
    )
    plt.close()

    print("\nTOP 10 PRODUCTS")
    print(top_products.sort_values(ascending=False))

    print("\nTOP 10 COUNTRIES")
    print(country_sales.head(10))

    print("\nMONTHLY REVENUE")
    print(monthly.to_string(index=False))

    print("\nCharts saved to:", FIGURE_DIR)


if __name__ == "__main__":
    run_business_eda()
