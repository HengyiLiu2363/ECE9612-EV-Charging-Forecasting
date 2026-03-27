from __future__ import annotations

# Handle file paths for the split datasets
from pathlib import Path

# Use pandas for loading data and simple time-series operations
import pandas as pd

# Use sklearn metrics to evaluate baseline performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Input files created from the chronological split step
TRAIN_FILE = Path("data/processed/train_hourly.csv")
VAL_FILE = Path("data/processed/val_hourly.csv")
TEST_FILE = Path("data/processed/test_hourly.csv")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load one split of the hourly forecasting dataset.
    """
    df = pd.read_csv(file_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    return df


def add_lag_1_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple lag-1 feature.

    For each hour, use the previous hour's session_count
    as the baseline prediction input.
    """
    df = df.copy()
    df["lag_1"] = df["session_count"].shift(1)
    return df


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series, dataset_name: str) -> None:
    """
    Print standard regression metrics for the baseline model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} results")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))


def main() -> None:
    """
    Baseline forecast:
    predict the current hour's session_count using the previous hour's session_count.
    """
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    test = load_data(TEST_FILE)

    train = add_lag_1_feature(train)
    val = add_lag_1_feature(val)
    test = add_lag_1_feature(test)

    # Drop the first row in each split because lag_1 is missing there
    train = train.dropna(subset=["lag_1"]).copy()
    val = val.dropna(subset=["lag_1"]).copy()
    test = test.dropna(subset=["lag_1"]).copy()

    # Baseline prediction is simply the lag_1 value
    train_pred = train["lag_1"]
    val_pred = val["lag_1"]
    test_pred = test["lag_1"]

    # True target is the actual session_count
    y_train = train["session_count"]
    y_val = val["session_count"]
    y_test = test["session_count"]

    evaluate_regression(y_train, train_pred, "Train")
    evaluate_regression(y_val, val_pred, "Validation")
    evaluate_regression(y_test, test_pred, "Test")


if __name__ == "__main__":
    main()