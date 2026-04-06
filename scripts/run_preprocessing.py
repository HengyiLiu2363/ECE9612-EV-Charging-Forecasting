"""
Legacy helper for generic CSV cleaning.

This script is not part of the final ACN forecasting pipeline, which uses:
- src/data/process_acn_data.py
- src/data/build_hourly_dataset.py
- src/models/split_data.py
- src/features/build_model_features.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_csv_data, preview_data
from src.data.preprocess import basic_cleaning


def main() -> None:
    print(
        "Legacy preprocessing helper: the final ACN workflow uses "
        "src/data/process_acn_data.py instead."
    )

    raw_path = Path("data/raw")
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(raw_path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in data/raw")
        return

    input_file = csv_files[0]
    print(f"Loading file: {input_file}")

    df = load_csv_data(str(input_file))
    preview_data(df)

    # Update these column names after inspecting the real dataset
    datetime_columns = []

    df_clean = basic_cleaning(df, datetime_columns=datetime_columns)

    output_file = processed_path / "cleaned_data.csv"
    df_clean.to_csv(output_file, index=False)

    print(f"\nCleaned data saved to: {output_file}")
    print(f"Cleaned shape: {df_clean.shape}")


if __name__ == "__main__":
    main()
