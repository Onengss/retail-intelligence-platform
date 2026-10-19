
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data/processed/daily_sales.csv"
REPORT_DIR = BASE_DIR / "reports"
ARTIFACT_DIR = BASE_DIR / "artifacts"

FEATURES = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    denominator = np.sum(np.abs(y_true))

    wape = (
        np.sum(np.abs(y_true - y_pred))
        / denominator
        if denominator > 0 else np.nan
    )

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "WAPE": round(float(wape), 4),
    }


def prepare_features(df):
    df = df.sort_values("date").copy()

    # Modelling assumption:
    # missing transaction dates represent zero
    # recorded revenue.
    df["target_revenue"] = df["revenue"].fillna(0)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Historical features: all use past revenue only.
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = (
            df["target_revenue"].shift(lag)
        )

    df["rolling_mean_7"] = (
        df["target_revenue"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["rolling_mean_28"] = (
        df["target_revenue"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    # Remove initial dates without enough history.
    df = df.dropna(subset=FEATURES).copy()

    return df


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    df = prepare_features(df)

    train = df[
        df["date"] <= "2011-06-30"
    ].copy()

    validation = df[
        (df["date"] > "2011-06-30")
        & (df["date"] <= "2011-09-30")
    ].copy()

    test = df[
        df["date"] > "2011-09-30"
    ].copy()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURES],
        train["target_revenue"],
    )

    print("=" * 60)
    print("RANDOM FOREST SALES FORECASTING")
    print("=" * 60)

    results = []
    predictions = []

    for split_name, subset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        y_true = subset["target_revenue"].to_numpy()
        y_pred = np.maximum(
            model.predict(subset[FEATURES]),
            0,
        )

        metrics = evaluate(y_true, y_pred)
        metrics["split"] = split_name
        metrics["days"] = len(subset)

        results.append(metrics)

        prediction_df = pd.DataFrame({
            "date": subset["date"],
            "split": split_name,
            "actual": y_true,
            "prediction": y_pred,
        })

        predictions.append(prediction_df)

        print(f"\n{split_name.upper()}")
        print(f"Days: {len(subset)}")
        print(f"MAE: £{metrics['MAE']:,.2f}")
        print(f"RMSE: £{metrics['RMSE']:,.2f}")
        print(f"WAPE: {metrics['WAPE']:.2%}")

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "random_forest_metrics.csv",
        index=False,
    )

    pd.concat(predictions).to_csv(
        REPORT_DIR / "random_forest_predictions.csv",
        index=False,
    )

    importance = (
        pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
    )

    importance.to_csv(
        REPORT_DIR / "forecast_feature_importance.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "forecast_type": "rolling_one_day_ahead",
        },
        ARTIFACT_DIR / "retail_forecast_model.joblib",
    )

    print("\nTOP FEATURES")
    print(importance.head(10).to_string(index=False))

    print("\nModel and reports saved successfully.")


if __name__ == "__main__":
    main()
