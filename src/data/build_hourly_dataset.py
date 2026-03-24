from __future__ import annotations

# Standard library imports
from pathlib import Path

# Third-party imports
import pandas as pd


# Input processed session-level file
INPUT_FILE = Path("data/processed/acn_sessions_flat.csv")

# Output hourly forecasting dataset
OUTPUT_FILE = Path("data/processed/hourly_demand.csv")


def load_sessions(file_path: Path) -> pd.DataFrame:
    """
    Load the processed session-level CSV.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Processed session file not found: {file_path}")
    return pd.read_csv(file_path)


def build_hourly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert session-level data into an hourly forecasting dataset.

    Output columns include:
    - hour
    - session_count
    - total_kwh
    - hour_of_day
    - day_of_week
    - month
    - is_weekend
    """
    df = df.copy()

    # Convert time columns back into datetime
    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True, errors="coerce")
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True, errors="coerce")
    df["doneChargingTime"] = pd.to_datetime(df["doneChargingTime"], utc=True, errors="coerce")

    # Keep only valid rows
    df = df[df["connectionTime"].notna()].copy()
    df = df[df["kWhDelivered"].notna()].copy()

    # Define the hour bin based on connection time
    df["hour"] = df["connectionTime"].dt.floor("h")

    # Aggregate sessions by hour
    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            session_count=("connectionTime", "count"),
            total_kwh=("kWhDelivered", "sum"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # Build a continuous hourly timeline so zero-demand hours are included
    full_hours = pd.date_range(
        start=hourly["hour"].min(),
        end=hourly["hour"].max(),
        freq="h",
        tz="UTC",
    )

    hourly = (
        pd.DataFrame({"hour": full_hours})
        .merge(hourly, on="hour", how="left")
        .fillna({"session_count": 0, "total_kwh": 0})
    )

    # Set proper numeric types
    hourly["session_count"] = hourly["session_count"].astype(int)
    hourly["total_kwh"] = hourly["total_kwh"].astype(float)

    # Create simple time-based features for forecasting
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["day_of_week"] = hourly["hour"].dt.dayofweek
    hourly["month"] = hourly["hour"].dt.month
    hourly["is_weekend"] = hourly["day_of_week"].isin([5, 6]).astype(int)

    return hourly


def main() -> None:
    """
    Main aggregation step:
    - load processed session data
    - aggregate into hourly demand
    - fill missing hours with zeros
    - save the final hourly forecasting dataset
    """
    df = load_sessions(INPUT_FILE)
    print(f"Loaded processed sessions: {df.shape}")

    hourly = build_hourly_dataset(df)
    print(f"Hourly dataset shape: {hourly.shape}")
    print("\nFirst rows:")
    print(hourly.head())
    print("\nLast rows:")
    print(hourly.tail())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved hourly dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()