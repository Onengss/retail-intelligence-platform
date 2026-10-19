
from pathlib import Path

import joblib
import mlflow
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports"
MODEL_FILE = BASE_DIR / "artifacts/retail_forecast_model.joblib"

EXPERIMENT_NAME = "retail-daily-sales-forecasting"


def main():
    # Store experiments locally in the project.
    tracking_db = BASE_DIR / "mlruns" / "mlflow.db"
    tracking_db.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    metrics_df = pd.read_csv(
        REPORT_DIR / "random_forest_metrics.csv"
    )
    test_metrics = metrics_df.loc[metrics_df["split"] == "test"]
    if len(test_metrics) != 1:
        raise ValueError(
            "Expected exactly one test split in random_forest_metrics.csv; "
            f"found {len(test_metrics)}."
        )
    test_metrics = test_metrics.iloc[0]

    model_bundle = joblib.load(MODEL_FILE)

    with mlflow.start_run(
        run_name="random_forest_daily_revenue_v1"
    ):
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 3,
            "random_state": 42,
            "forecast_horizon": "rolling_one_day_ahead",
            "missing_revenue_policy": "fill_zero",
            "target": "daily_merchandise_revenue_gbp",
            "feature_count": len(
                model_bundle["features"]
            ),
            "train_end": "2011-06-30",
            "validation_end": "2011-09-30",
        })

        for _, row in metrics_df.iterrows():
            split = row["split"]

            mlflow.log_metric(
                f"{split}_mae",
                float(row["MAE"])
            )
            mlflow.log_metric(
                f"{split}_rmse",
                float(row["RMSE"])
            )
            mlflow.log_metric(
                f"{split}_wape",
                float(row["WAPE"])
            )

        for filename in [
            "random_forest_metrics.csv",
            "random_forest_predictions.csv",
            "forecast_feature_importance.csv",
            "baseline_metrics.csv",
        ]:
            path = REPORT_DIR / filename
            if path.exists():
                mlflow.log_artifact(
                    str(path),
                    artifact_path="reports"
                )

        mlflow.log_artifact(
            str(MODEL_FILE),
            artifact_path="model"
        )

        mlflow.set_tags({
            "project": "retail-intelligence-platform",
            "dataset": "UCI Online Retail II",
            "model_stage": "experimental",
            "evaluation": "chronological_split",
        })

        print("=" * 60)
        print("MLFLOW EXPERIMENT REGISTERED")
        print("=" * 60)
        print("Experiment:", EXPERIMENT_NAME)
        print("Run ID:", mlflow.active_run().info.run_id)
        print(f"Test MAE: {test_metrics['MAE']:,.2f} GBP")
        print(f"Test WAPE: {test_metrics['WAPE']:.2%}")
        print("Tracking database:", tracking_db)


if __name__ == "__main__":
    main()
