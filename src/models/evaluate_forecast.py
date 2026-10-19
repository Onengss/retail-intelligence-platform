
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rf = pd.read_csv(
        REPORT_DIR / "random_forest_predictions.csv",
        parse_dates=["date"],
    )

    baseline = pd.read_csv(
        REPORT_DIR / "baseline_predictions.csv",
        parse_dates=["date"],
    )

    rf_test = (
        rf[rf["split"] == "test"]
        .sort_values("date")
        .copy()
    )

    baseline_test = (
        baseline[baseline["split"] == "test"]
        [["date", "prediction"]]
        .rename(columns={
            "prediction": "baseline_prediction"
        })
    )

    test = rf_test.merge(
        baseline_test,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    if len(test) != len(rf_test):
        raise ValueError(
            "Baseline and Random Forest test dates do not match"
        )

    test["rf_abs_error"] = (
        test["actual"] - test["prediction"]
    ).abs()

    test["baseline_abs_error"] = (
        test["actual"] - test["baseline_prediction"]
    ).abs()

    test["signed_error"] = (
        test["prediction"] - test["actual"]
    )

    # Chart 1: Actual versus predicted revenue
    plt.figure(figsize=(14, 6))

    plt.plot(
        test["date"],
        test["actual"],
        label="Actual Revenue",
        linewidth=2,
    )

    plt.plot(
        test["date"],
        test["prediction"],
        label="Random Forest",
        alpha=0.85,
    )

    plt.plot(
        test["date"],
        test["baseline_prediction"],
        label="Seasonal Naive",
        alpha=0.7,
        linestyle="--",
    )

    plt.title("Daily Revenue Forecast — Test Period")
    plt.xlabel("Date")
    plt.ylabel("Revenue (GBP)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "forecast_actual_vs_predicted.png",
        dpi=160,
    )
    plt.close()

    # Chart 2: Error distribution
    plt.figure(figsize=(10, 5))
    plt.hist(test["signed_error"], bins=20)
    plt.axvline(0, linestyle="--", color="black")
    plt.title("Random Forest Forecast Error Distribution")
    plt.xlabel("Predicted - Actual Revenue (GBP)")
    plt.ylabel("Number of Days")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "forecast_error_distribution.png",
        dpi=160,
    )
    plt.close()

    # Chart 3: Absolute error comparison
    plt.figure(figsize=(14, 5))

    plt.plot(
        test["date"],
        test["baseline_abs_error"],
        label="Seasonal Naive",
        alpha=0.7,
    )

    plt.plot(
        test["date"],
        test["rf_abs_error"],
        label="Random Forest",
    )

    plt.title("Daily Absolute Forecast Errors")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error (GBP)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "forecast_absolute_errors.png",
        dpi=160,
    )
    plt.close()

    # Top 10 largest Random Forest errors
    worst_days = (
        test.sort_values(
            "rf_abs_error",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    worst_days["weekday"] = (
        worst_days["date"].dt.day_name()
    )

    worst_days.to_csv(
        REPORT_DIR / "forecast_worst_days.csv",
        index=False,
    )

    test.to_csv(
        REPORT_DIR / "forecast_test_evaluation.csv",
        index=False,
    )

    print("=" * 60)
    print("FORECAST ERROR ANALYSIS")
    print("=" * 60)

    print(f"Test days: {len(test)}")

    print(
        "Random Forest MAE:",
        round(test["rf_abs_error"].mean(), 2)
    )

    print(
        "Seasonal Naive MAE:",
        round(test["baseline_abs_error"].mean(), 2)
    )

    print(
        "Mean signed error:",
        round(test["signed_error"].mean(), 2)
    )

    print(
        "\nRandom Forest better days:",
        int(
            (
                test["rf_abs_error"]
                < test["baseline_abs_error"]
            ).sum()
        )
    )

    print(
        "\nTOP 10 LARGEST ERRORS"
    )

    print(
        worst_days[
            [
                "date",
                "weekday",
                "actual",
                "prediction",
                "rf_abs_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\nCharts saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
