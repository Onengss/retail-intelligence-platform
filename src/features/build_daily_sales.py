
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data/processed/retail_merchandise.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/daily_sales.csv"


def build_daily_sales():
    print("=" * 60)
    print("BUILDING DAILY SALES DATASET")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE,
        usecols=["InvoiceDate", "Revenue"],
        parse_dates=["InvoiceDate"],
    )

    df["Date"] = df["InvoiceDate"].dt.normalize()

    daily = (
        df.groupby("Date")["Revenue"]
        .sum()
        .sort_index()
        .rename("revenue")
        .to_frame()
    )

    # Create a complete daily calendar.
    # Missing dates are flagged rather than silently
    # treated as genuine zero-sales days.
    calendar = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D",
    )

    daily = daily.reindex(calendar)
    daily.index.name = "date"

    daily["has_transactions"] = (
        daily["revenue"].notna()
    ).astype(int)

    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["year"] = daily.index.year

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    daily.to_csv(OUTPUT_FILE)

    print(f"Calendar days: {len(daily):,}")
    print(
        "Days with transactions:",
        daily["has_transactions"].sum()
    )
    print(
        "Days without transactions:",
        (daily["has_transactions"] == 0).sum()
    )
    print(
        "Total recorded revenue: £",
        round(daily["revenue"].sum(), 2)
    )
    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    build_daily_sales()
