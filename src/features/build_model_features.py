from __future__ import annotations

# Handle file paths
from pathlib import Path

# Use pandas for time-series feature construction
import pandas as pd


# Input files from the split step
TRAIN_FILE = Path("data/processed/train_hourly.csv")
VAL_FILE = Path("data/processed/val_hourly.csv")
TEST_FILE = Path("data/processed/test_hourly.csv")

# Output files for model-ready feature datasets
OUTPUT_DIR = Path("data/processed")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load one split of the hourly demand dataset.
    """
    df = pd.read_csv(file_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple calendar-based features from the hour timestamp.
    """
    df = df.copy()

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["month"] = df["hour"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def add_session_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag-based features from historical session_count.
    """
    df = df.copy()

    df["session_lag_1"] = df["session_count"].shift(1)
    df["session_lag_2"] = df["session_count"].shift(2)
    df["session_lag_24"] = df["session_count"].shift(24)
    df["session_lag_168"] = df["session_count"].shift(168)
    df["session_rolling_mean_24"] = df["session_count"].shift(1).rolling(24).mean()

    return df


def add_kwh_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag-based features from historical total_kwh.
    """
    df = df.copy()

    df["kwh_lag_1"] = df["total_kwh"].shift(1)
    df["kwh_lag_2"] = df["total_kwh"].shift(2)
    df["kwh_lag_24"] = df["total_kwh"].shift(24)
    df["kwh_lag_168"] = df["total_kwh"].shift(168)
    df["kwh_rolling_mean_24"] = df["total_kwh"].shift(1).rolling(24).mean()

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature table used by the forecasting models.
    """
    df = add_time_features(df)
    df = add_session_lag_features(df)
    df = add_kwh_lag_features(df)

    # Drop rows where lag features are missing
    df = df.dropna().copy()

    return df


def main() -> None:
    """
    Build model-ready train, validation, and test datasets.
    """
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    test = load_data(TEST_FILE)

    train_feat = prepare_features(train)
    val_feat = prepare_features(val)
    test_feat = prepare_features(test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_feat.to_csv(OUTPUT_DIR / "train_features.csv", index=False)
    val_feat.to_csv(OUTPUT_DIR / "val_features.csv", index=False)
    test_feat.to_csv(OUTPUT_DIR / "test_features.csv", index=False)

    print("Train feature shape:", train_feat.shape)
    print("Validation feature shape:", val_feat.shape)
    print("Test feature shape:", test_feat.shape)

    print("\nFeature columns:")
    print(train_feat.columns.tolist())


if __name__ == "__main__":
    main()