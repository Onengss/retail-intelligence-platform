
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data/processed/daily_sales.csv"
REPORT_DIR = BASE_DIR / "reports"


def evaluate(y_true, y_pred):
    error = y_true - y_pred

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    denominator = np.sum(np.abs(y_true))
    wape = (
        np.sum(np.abs(error)) / denominator
        if denominator > 0 else np.nan
    )

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "WAPE": round(float(wape), 4),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    ).sort_values("date")

    # Assumption: no recorded transactions = zero
    # recorded revenue for this modelling experiment.
    df["target_revenue"] = df["revenue"].fillna(0)

    # Seasonal naive forecast:
    # same weekday from the previous week.
    df["prediction"] = (
        df["target_revenue"].shift(7)
    )

    # Chronological split
    train_end = pd.Timestamp("2011-06-30")
    valid_end = pd.Timestamp("2011-09-30")

    df["split"] = np.select(
        [
            df["date"] <= train_end,
            df["date"] <= valid_end,
        ],
        ["train", "validation"],
        default="test",
    )

    print("=" * 60)
    print("7-DAY SEASONAL NAIVE BASELINE")
    print("=" * 60)

    results = []

    for split in ["train", "validation", "test"]:
        subset = df[
            (df["split"] == split)
            & df["prediction"].notna()
        ].copy()

        metrics = evaluate(
            subset["target_revenue"].to_numpy(),
            subset["prediction"].to_numpy(),
        )

        metrics["split"] = split
        metrics["days"] = len(subset)

        results.append(metrics)

        print(f"\n{split.upper()}")
        print(f"Days: {len(subset)}")
        print(f"MAE: £{metrics['MAE']:,.2f}")
        print(f"RMSE: £{metrics['RMSE']:,.2f}")
        print(f"WAPE: {metrics['WAPE']:.2%}")

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "baseline_metrics.csv",
        index=False
    )

    df.to_csv(
        REPORT_DIR / "baseline_predictions.csv",
        index=False
    )

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()
