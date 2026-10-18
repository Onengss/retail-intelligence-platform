
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

load_dotenv(BASE_DIR / ".env")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "retail_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "retail_intelligence"),
    )

    engine = create_engine(url)

    query = """
        SELECT customer_id, recency, frequency,
               monetary, r_score, f_score, m_score,
               rfm_total, segment
        FROM customer_rfm
    """

    with engine.connect() as conn:
        rfm = pd.read_sql_query(query, conn)

    print("=" * 60)
    print("RFM CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 60)
    print(f"Customers: {len(rfm):,}")

    summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_spend=("monetary", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )

    summary["customer_share_pct"] = (
        summary["customers"] / len(rfm) * 100
    )
    summary["revenue_share_pct"] = (
        summary["revenue"] / summary["revenue"].sum() * 100
    )

    summary.to_csv(REPORT_DIR / "rfm_segment_summary.csv")
    rfm.to_csv(REPORT_DIR / "rfm_customer_scores.csv", index=False)

    print("\nSEGMENT SUMMARY")
    print(summary.round(2).to_string())

    # Chart 1: Customer distribution
    plt.figure(figsize=(10, 5))
    summary["customers"].sort_values().plot(kind="barh")
    plt.title("Customers by RFM Segment")
    plt.xlabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "rfm_customer_distribution.png",
        dpi=160
    )
    plt.close()

    # Chart 2: Revenue contribution
    plt.figure(figsize=(10, 5))
    summary["revenue_share_pct"].sort_values().plot(
        kind="barh"
    )
    plt.title("Identified Revenue Share by RFM Segment")
    plt.xlabel("Revenue Share (%)")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "rfm_revenue_share.png",
        dpi=160
    )
    plt.close()

    # Chart 3: Recency versus frequency
    plt.figure(figsize=(10, 6))

    for segment, group in rfm.groupby("segment"):
        plt.scatter(
            group["recency"],
            group["frequency"],
            label=segment,
            alpha=0.4,
            s=12,
        )

    plt.yscale("log")
    plt.title("Customer Recency vs Purchase Frequency")
    plt.xlabel("Recency (Days)")
    plt.ylabel("Orders (Log Scale)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "rfm_recency_frequency.png",
        dpi=160
    )
    plt.close()

    # Sensitivity: remove the audited exceptional line
    # from the existing customer's monetary value only.
    adjusted = rfm.copy()
    anomaly_customer = "16446.0"
    anomaly_revenue = 168469.60

    mask = (
        adjusted["customer_id"].astype(str)
        == anomaly_customer
    )

    if mask.sum() != 1:
        raise ValueError(
            "Expected exactly one matching anomaly customer"
        )

    adjusted.loc[mask, "monetary"] -= anomaly_revenue

    comparison = pd.DataFrame({
        "Original": rfm.groupby("segment")["monetary"].sum(),
        "Adjusted": adjusted.groupby("segment")["monetary"].sum(),
    }).fillna(0)

    comparison.to_csv(
        REPORT_DIR / "rfm_anomaly_sensitivity.csv"
    )

    print("\nREVENUE SENSITIVITY ANALYSIS")
    print(comparison.round(2).to_string())
    print(
        "\nNote: This comparison holds segment labels fixed."
        "\nIt is not a full RFM re-segmentation."
    )

    print("\nFigures saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
