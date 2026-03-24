from __future__ import annotations

# Handle file paths in a clean cross-platform way
from pathlib import Path

# Use pandas for reading and splitting the time-series dataset
import pandas as pd


# This is the final hourly forecasting dataset created from the ACN sessions
INPUT_FILE = Path("data/processed/hourly_demand.csv")

# Save the split datasets here so later model scripts can load them directly
OUTPUT_DIR = Path("data/processed")


def main() -> None:
    """
    Split the hourly demand dataset into train, validation, and test sets.

    The split is chronological because this is a forecasting problem.
    We must train on earlier data and evaluate on later data.
    """
    # Load the hourly demand dataset
    df = pd.read_csv(INPUT_FILE)

    # Convert the hour column from string to datetime format
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")

    # Training data: used to fit the models
    train = df[(df["hour"] >= "2019-01-01") & (df["hour"] < "2021-06-01")].copy()

    # Validation data: used to compare models and tune the approach
    val = df[(df["hour"] >= "2021-06-01") & (df["hour"] < "2021-08-01")].copy()

    # Test data: used only for the final performance evaluation
    test = df[(df["hour"] >= "2021-08-01") & (df["hour"] <= "2021-09-14 23:59:59")].copy()

    # Create the output folder if it does not already exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save each split so later forecasting scripts can use the same data consistently
    train.to_csv(OUTPUT_DIR / "train_hourly.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val_hourly.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_hourly.csv", index=False)

    # Print shapes to verify how many rows are in each split
    print("Train shape:", train.shape)
    print("Validation shape:", val.shape)
    print("Test shape:", test.shape)

    # Print time ranges to confirm the split boundaries are correct
    print("\nTrain range:", train["hour"].min(), "to", train["hour"].max())
    print("Validation range:", val["hour"].min(), "to", val["hour"].max())
    print("Test range:", test["hour"].min(), "to", test["hour"].max())


if __name__ == "__main__":
    main()