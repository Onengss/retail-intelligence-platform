
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
DATA_SOURCE = os.getenv("DATA_SOURCE", "csv").lower()
DASHBOARD_DATA_DIR = BASE_DIR / "dashboard" / "data"

st.set_page_config(
    page_title="Retail Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

st.sidebar.markdown("## Retail Intelligence")
st.sidebar.caption("Data Science Portfolio Project")
st.sidebar.divider()


@st.cache_resource
def get_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "retail_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv(
            "POSTGRES_DB", "retail_intelligence"
        ),
    )
    return create_engine(url)


@st.cache_data(ttl=600)
def load_overview():
    if DATA_SOURCE == "csv":
        kpi = pd.read_csv(
            DASHBOARD_DATA_DIR / "overview_kpi.csv"
        )
        monthly = pd.read_csv(
            DASHBOARD_DATA_DIR / "overview_monthly.csv",
            parse_dates=["month"],
        )
        countries = pd.read_csv(
            DASHBOARD_DATA_DIR / "overview_countries.csv"
        )
        return kpi, monthly, countries

    engine = get_engine()

    kpi_query = """
        SELECT
            COUNT(DISTINCT invoice) AS orders,
            COUNT(DISTINCT customer_id) AS customers,
            ROUND(SUM(revenue), 2) AS revenue,
            SUM(quantity) AS units
        FROM retail_transactions
    """

    monthly_query = """
        SELECT
            DATE_TRUNC('month', invoice_date)::date AS month,
            SUM(revenue)::float AS revenue
        FROM retail_transactions
        GROUP BY 1
        ORDER BY 1
    """

    country_query = """
        SELECT
            country,
            SUM(revenue)::float AS revenue
        FROM retail_transactions
        GROUP BY country
        ORDER BY revenue DESC
        LIMIT 10
    """

    with engine.connect() as conn:
        kpi = pd.read_sql_query(kpi_query, conn)
        monthly = pd.read_sql_query(
            monthly_query, conn
        )
        countries = pd.read_sql_query(
            country_query, conn
        )

    return kpi, monthly, countries


@st.cache_data(ttl=600)
def load_customer_segments():
    if DATA_SOURCE == "csv":
        segments = pd.read_csv(
            DASHBOARD_DATA_DIR / "customer_segments.csv"
        )
    else:
        engine = get_engine()
        query = """
            SELECT
                segment,
                COUNT(*) AS customers,
                SUM(monetary)::float AS revenue,
                AVG(recency)::float AS avg_recency,
                AVG(frequency)::float AS avg_frequency,
                AVG(monetary)::float AS avg_spend
            FROM customer_rfm
            GROUP BY segment
            ORDER BY revenue DESC
        """

        with engine.connect() as conn:
            segments = pd.read_sql_query(query, conn)
        engine.dispose()

    if segments.empty:
        return segments

    customer_total = segments["customers"].sum()
    revenue_total = segments["revenue"].sum()
    segments["customer_share_pct"] = (
        segments["customers"] / customer_total * 100
    )
    segments["revenue_share_pct"] = (
        segments["revenue"] / revenue_total * 100
    )
    return segments


@st.cache_data(ttl=600)
def load_forecasting_data():
    reports = BASE_DIR / "reports"
    baseline = pd.read_csv(
        reports / "baseline_metrics.csv"
    )
    rf_metrics = pd.read_csv(
        reports / "random_forest_metrics.csv"
    )
    predictions = pd.read_csv(
        reports / "forecast_test_evaluation.csv",
        parse_dates=["date"],
    )
    importance = pd.read_csv(
        reports / "forecast_feature_importance.csv"
    )
    worst_days = pd.read_csv(
        reports / "forecast_worst_days.csv",
        parse_dates=["date"],
    )
    return (
        baseline,
        rf_metrics,
        predictions,
        importance,
        worst_days,
    )


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Customer Segmentation",
        "Sales Forecasting",
    ],
)

if page == "Sales Overview":
    st.title("Retail Intelligence Platform")
    st.caption(
        "Business intelligence, customer analytics "
        "and machine learning forecasting"
    )

    st.info(
        "Historical UCI Online Retail II data (2009–2011). "
        "Revenue represents positive merchandise sales "
        "before returns, in GBP."
    )

    try:
        kpi, monthly, countries = load_overview()
    except Exception as exc:
        st.error(f"Unable to load Sales Overview data from {DATA_SOURCE}.")
        st.exception(exc)
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Merchandise Revenue",
        f"£{float(kpi.loc[0, 'revenue']):,.2f}"
    )

    col2.metric(
        "Orders",
        f"{int(kpi.loc[0, 'orders']):,}"
    )

    col3.metric(
        "Known Customers",
        f"{int(kpi.loc[0, 'customers']):,}"
    )

    col4.metric(
        "Units Sold",
        f"{int(kpi.loc[0, 'units']):,}"
    )

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Monthly Revenue")
        fig = px.line(
            monthly,
            x="month",
            y="revenue",
            markers=True,
            labels={
                "month": "Month",
                "revenue": "Revenue (GBP)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "December 2011 is incomplete. "
            "Monthly totals are not adjusted "
            "for differences in trading days."
        )

    with right:
        st.subheader("Top 10 Countries")
        fig = px.bar(
            countries.sort_values("revenue"),
            x="revenue",
            y="country",
            orientation="h",
            labels={
                "revenue": "Revenue (GBP)",
                "country": "Country",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Project Modules")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### Sales Analytics")
        st.write(
            "Revenue trends, product performance "
            "and geographical sales."
        )

    with b:
        st.markdown("### Customer Segmentation")
        st.write(
            "RFM scores, customer groups "
            "and revenue contribution."
        )

    with c:
        st.markdown("### Sales Forecasting")
        st.write(
            "Machine learning forecasts, "
            "baseline comparisons and error analysis."
        )

elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.caption(
        "RFM Analysis — Recency, Frequency, Monetary Value"
    )
    st.info(
        "Customer segments are based on historical "
        "positive merchandise purchases. "
        "Transactions without customer IDs are excluded."
    )

    try:
        segments = load_customer_segments()
    except Exception as exc:
        st.error("Unable to load customer segmentation data.")
        st.exception(exc)
        st.stop()

    if segments.empty:
        st.warning("No customer segment data is available.")
        st.stop()

    key_segments = (
        segments.set_index("segment")
        .reindex(["Champions", "At Risk"])
        .fillna(0)
    )
    champions = key_segments.loc["Champions"]
    at_risk = key_segments.loc["At Risk"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Identified Customers",
        f"{int(segments['customers'].sum()):,}"
    )
    c2.metric(
        "Champions",
        f"{int(champions['customers']):,}"
    )
    c3.metric(
        "Champion Revenue Share",
        f"{float(champions['revenue_share_pct']):.1f}%"
    )
    c4.metric(
        "At Risk Customers",
        f"{int(at_risk['customers']):,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customers by Segment")
        fig = px.bar(
            segments.sort_values("customers"),
            x="customers",
            y="segment",
            orientation="h",
            labels={
                "customers": "Customers",
                "segment": "Segment",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Revenue Share by Segment")
        fig = px.bar(
            segments.sort_values("revenue_share_pct"),
            x="revenue_share_pct",
            y="segment",
            orientation="h",
            labels={
                "revenue_share_pct": "Revenue Share (%)",
                "segment": "Segment",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Summary")
    summary = segments.rename(
        columns={
            "segment": "Segment",
            "customers": "Customers",
            "customer_share_pct": "Customer Share (%)",
            "revenue": "Revenue (GBP)",
            "revenue_share_pct": "Revenue Share (%)",
            "avg_recency": "Avg Recency (days)",
            "avg_frequency": "Avg Frequency",
            "avg_spend": "Avg Spend (GBP)",
        }
    )
    st.dataframe(
        summary[
            [
                "Segment",
                "Customers",
                "Customer Share (%)",
                "Revenue (GBP)",
                "Revenue Share (%)",
                "Avg Recency (days)",
                "Avg Frequency",
                "Avg Spend (GBP)",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

elif page == "Sales Forecasting":
    st.title("Sales Forecasting")
    st.caption(
        "Daily revenue forecast evaluation — Random Forest "
        "compared with a seasonal naive baseline"
    )
    st.info(
        "This page displays previously generated evaluation "
        "reports. Opening the dashboard does not retrain the model."
    )

    try:
        (
            baseline,
            rf_metrics,
            predictions,
            importance,
            worst_days,
        ) = load_forecasting_data()
    except Exception as exc:
        st.error(
            "Unable to load forecasting reports. "
            "Run the forecasting pipeline to generate them."
        )
        st.exception(exc)
        st.stop()

    baseline_test = baseline.loc[baseline["split"] == "test"]
    rf_test = rf_metrics.loc[rf_metrics["split"] == "test"]
    if baseline_test.empty or rf_test.empty:
        st.error("Forecast reports do not contain test-split metrics.")
        st.stop()
    if predictions.empty or importance.empty or worst_days.empty:
        st.warning(
            "One or more forecasting reports contain no data."
        )
        st.stop()

    baseline_test = baseline_test.iloc[0]
    rf_test = rf_test.iloc[0]
    mae_improvement = (
        (baseline_test["MAE"] - rf_test["MAE"])
        / baseline_test["MAE"]
        * 100
    )
    comparison = pd.DataFrame(
        [
            {
                "Model": "Seasonal Naive",
                "MAE": baseline_test["MAE"],
                "RMSE": baseline_test["RMSE"],
                "WAPE": baseline_test["WAPE"],
            },
            {
                "Model": "Random Forest",
                "MAE": rf_test["MAE"],
                "RMSE": rf_test["RMSE"],
                "WAPE": rf_test["WAPE"],
            },
        ]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Test Days",
        f"{len(predictions):,}",
    )
    c2.metric(
        "Random Forest MAE",
        f"£{rf_test['MAE']:,.2f}",
    )
    c3.metric(
        "Seasonal Naive MAE",
        f"£{baseline_test['MAE']:,.2f}",
    )
    c4.metric(
        "MAE Improvement",
        f"{mae_improvement:.2f}%",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Model Comparison")
        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "MAE": st.column_config.NumberColumn(
                    "MAE (GBP)",
                    format="£%.2f",
                ),
                "RMSE": st.column_config.NumberColumn(
                    "RMSE (GBP)",
                    format="£%.2f",
                ),
                "WAPE": st.column_config.NumberColumn(
                    "WAPE",
                    format="%.4f",
                ),
            },
        )
        fig = px.bar(
            comparison,
            x="Model",
            y="MAE",
            color="Model",
            title="Test MAE Comparison",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Feature Importance")
        fig = px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            labels={
                "importance": "Importance",
                "feature": "Feature",
            },
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Actual vs Predicted Revenue")
    prediction_series = predictions.melt(
        id_vars=["date"],
        value_vars=["actual", "prediction"],
        var_name="series",
        value_name="revenue",
    )
    fig = px.line(
        prediction_series,
        x="date",
        y="revenue",
        color="series",
        labels={
            "date": "Date",
            "revenue": "Revenue (GBP)",
            "series": "Series",
        },
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Largest Forecasting Errors")
    display_errors = worst_days[
        [
            "date",
            "weekday",
            "actual",
            "prediction",
            "rf_abs_error",
        ]
    ].copy()
    st.dataframe(
        display_errors,
        hide_index=True,
        width="stretch",
        column_config={
            "date": st.column_config.DateColumn(
                "Date",
                format="YYYY-MM-DD",
            ),
            "actual": st.column_config.NumberColumn(
                "Actual Revenue",
                format="£%.2f",
            ),
            "prediction": st.column_config.NumberColumn(
                "Predicted Revenue",
                format="£%.2f",
            ),
            "rf_abs_error": st.column_config.NumberColumn(
                "Absolute Error",
                format="£%.2f",
            ),
        },
    )

    st.divider()
    st.subheader("Key Business Insights")
    st.markdown(
        """
        **1. Improved forecasting accuracy**
        Random Forest reduced test MAE by approximately
        18.74% compared with the 7-day seasonal naive
        baseline.

        **2. Weekly patterns matter**
        The 7-day rolling mean and previous-week
        revenue were the strongest model features.

        **3. Large transactions create uncertainty**
        A £168,469.60 transaction on December 9, 2011
        represented 85.04% of that day's recorded revenue.

        **4. Model limitations**
        Results are based on historical one-day-ahead
        predictions. The model has not been validated
        for live deployment or multi-day forecasting.
        """
    )
