# Retail Intelligence Platform

### End-to-End Data Analytics, Customer Segmentation & Sales Forecasting

A data science portfolio project that transforms historical retail transactions into actionable business insights using **Python, PostgreSQL, SQL, scikit-learn, MLflow, Streamlit, and Plotly**.

The project demonstrates an end-to-end analytical workflow, from data ingestion and quality checks to customer segmentation, machine learning, model evaluation, and interactive dashboard development.

| Transactions | Identified customers | Test MAE improvement | Dashboard |
| --- | --- | --- | --- |
| 1.04M merchandise lines | 5,862 | 18.74% | Streamlit |

## Project Highlights

| Metric | Result |
| --- | ---: |
| Original dataset | 1,067,371 transaction lines |
| Cleaned merchandise dataset | 1,037,428 transaction lines |
| Positive merchandise revenue | £20,147,064.97 |
| Distinct orders | 39,581 |
| Identified customers | 5,862 |
| Random Forest test MAE | £11,438.86 |
| Seasonal naive test MAE | £14,077.42 |
| Test MAE improvement | 18.74% |
| Random Forest test WAPE | 25.16% |

## Business Objectives

1. Understand retail sales performance across products, countries, and time.
2. Identify valuable customer groups through RFM segmentation.
3. Develop and evaluate a daily revenue forecasting model.
4. Investigate unusual transactions and their impact on business reporting and model performance.
5. Present insights through an interactive business intelligence dashboard.

## Technology Stack

- **Data engineering:** Python, Pandas, PostgreSQL, Docker
- **Analytics:** SQL, Pandas, Matplotlib, Plotly
- **Machine learning:** scikit-learn, Random Forest Regressor
- **Experiment tracking:** MLflow
- **Dashboard:** Streamlit
- **Version control:** Git and GitHub

## Data Pipeline

```text
UCI Online Retail II dataset
    → Data ingestion
    → Data quality checks and cleaning
    → PostgreSQL storage
    → SQL business analytics
    → RFM customer segmentation
    → Daily revenue feature engineering
    → Seasonal naive baseline and Random Forest
    → Model evaluation and MLflow tracking
    → Streamlit dashboard
```

## Customer Segmentation

Customers were grouped using **Recency, Frequency, and Monetary (RFM)** scoring. Quintile scores are calculated from positive merchandise purchases; transactions without a customer ID are excluded.

| Segment | Customers | Customer share | Revenue share |
| --- | ---: | ---: | ---: |
| Champions | 1,286 | 21.94% | 68.17% |
| Loyal Customers | 1,400 | 23.88% | 15.24% |
| At Risk | 830 | 14.16% | 9.40% |
| Other Customers | 1,900 | 32.41% | 4.81% |
| New or Occasional | 446 | 7.61% | 2.37% |

Revenue shares are calculated over identified-customer merchandise spending, excluding returns. **Champions make up 21.94% of identified customers and contribute 68.17% of identified-customer revenue.**

## Sales Forecasting

Two forecasting approaches were evaluated:

- **Seasonal naive:** Predicts daily revenue using the same weekday from the previous week.
- **Random Forest:** Predicts daily revenue using historical lags, rolling averages, and calendar features.

### Test Performance

| Metric | Seasonal naive | Random Forest |
| --- | ---: | ---: |
| MAE | £14,077.42 | £11,438.86 |
| RMSE | £23,990.87 | £20,512.77 |
| WAPE | 30.97% | 25.16% |

Random Forest reduced test MAE by **18.74%** compared with the seasonal naive baseline. Evaluation uses chronological train, validation, and test splits. Test predictions use a rolling one-day-ahead setup.

The strongest Random Forest features were the **7-day rolling mean** and **revenue from the previous week** (`lag_7`).

## Data Quality and Anomaly Investigation

An exceptional merchandise line on December 9, 2011, recorded **80,995 units** of *PAPER CRAFT, LITTLE BIRDIE* at £2.08 each, for a line total of **£168,469.60**. This was **85.04%** of that day's recorded merchandise revenue and contributed to the largest test-period forecasting error.

The transaction was retained in the primary analysis because its validity could not be conclusively established. A sensitivity analysis was performed to understand its impact on customer segmentation.

## Interactive Dashboard

The Streamlit dashboard provides three pages:

- **Sales Overview:** Revenue KPIs, monthly sales trends, and country-level sales.
- **Customer Segmentation:** RFM segment distribution, revenue contribution, and a segment summary.
- **Sales Forecasting:** Model comparison, feature importance, actual versus predicted revenue, and forecasting error analysis.

The dashboard reads pre-generated forecasting reports and does not retrain the model when opened.

## Getting Started

### Prerequisites

- Python 3.11 or later
- Docker Desktop with Docker Compose
- The UCI Online Retail II workbook, saved as `data/raw/online_retail_II.xlsx`

The raw dataset is not included in this repository. Obtain it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Create the ignored data directory and place the downloaded workbook there:

```powershell
New-Item -ItemType Directory -Force data\raw
```

### Install dependencies

From the repository root, create and activate a virtual environment, then install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file with the PostgreSQL password used by Docker Compose and the dashboard:

```dotenv
POSTGRES_PASSWORD=replace-with-a-local-password
```

Other connection settings default to `localhost:5433`, database `retail_intelligence`, and user `retail_user`. Override them in `.env` if needed.

### Run the pipeline

Start PostgreSQL and create the cleaned datasets and database table:

```powershell
docker compose up -d postgres
python -m src.data.ingest
python -m src.data.clean
Get-Content sql\01_create_tables.sql | docker compose exec -T postgres psql -U retail_user -d retail_intelligence
python -m src.data.load_postgres
Get-Content sql\03_rfm_analysis.sql | docker compose exec -T postgres psql -U retail_user -d retail_intelligence
```

Build the daily sales data, train and evaluate the forecasts, and log the experiment:

```powershell
python -m src.features.build_daily_sales
python -m src.models.baseline_forecast
python -m src.models.train_forecast
python -m src.models.evaluate_forecast
python -m src.models.track_forecast
```

Launch the dashboard after the database and report files have been generated:

```powershell
python -m streamlit run dashboard\app.py
```

## Model Limitations

- The dataset covers December 2009 through December 2011.
- Merchandise revenue reflects positive sales before returns, not accounting net revenue.
- Dates without recorded transactions are treated as zero recorded revenue for this forecasting experiment.
- Evaluation uses rolling one-day-ahead predictions; the model has not been independently validated for multi-day forecasting.
- Unusual high-value transactions can cause substantial prediction errors.
- The model has not been validated for production deployment.

## Future Improvements

- Automate model retraining and MLflow logging.
- Expand anomaly handling and sensitivity testing.
- Evaluate additional forecasting models with time-series cross-validation.
- Deploy the dashboard with a reproducible data backend.
- Add automated data validation and tests.

## Author

**Muhamad Fauzan Bin Khairil Azhar**

- GitHub: [Onengss](https://github.com/Onengss)
- LinkedIn: [Muhamad Fauzan](https://www.linkedin.com/in/muhamad-fauzan-96b5b4263/)
