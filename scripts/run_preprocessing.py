from pathlib import Path

from src.data.load_data import load_csv_data, preview_data
from src.data.preprocess import basic_cleaning


def main() -> None:
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