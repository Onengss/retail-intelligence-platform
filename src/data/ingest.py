
from pathlib import Path
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Input and output locations
RAW_FILE = BASE_DIR / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "interim" / "retail_combined.csv"


def load_retail_data():
    print("Loading retail dataset...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    # Load every worksheet from the Excel workbook
    sheets = pd.read_excel(RAW_FILE, sheet_name=None)

    print(f"Worksheets found: {list(sheets.keys())}")

    # Combine both years into one dataset
    df = pd.concat(
        sheets.values(),
        ignore_index=True
    )

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Inspect the first five records
    print("\nFirst 5 rows:")
    print(df.head())

    # Inspect column names and types
    print("\nColumn data types:")
    print(df.dtypes)

    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save combined raw data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDataset saved to: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    load_retail_data()
