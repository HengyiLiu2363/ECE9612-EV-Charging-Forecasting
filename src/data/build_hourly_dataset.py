from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/acn_sessions_flat.csv")
OUTPUT_FILE = Path("data/processed/hourly_demand.csv")


def load_sessions(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Processed session file not found: {file_path}")
    return pd.read_csv(file_path)


def build_hourly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True, errors="coerce")
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True, errors="coerce")
    df["doneChargingTime"] = pd.to_datetime(df["doneChargingTime"], utc=True, errors="coerce")

    df = df[df["connectionTime"].notna()].copy()
    df = df[df["kWhDelivered"].notna()].copy()

    df["hour"] = df["connectionTime"].dt.floor("h")

    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            session_count=("connectionTime", "count"),
            total_kwh=("kWhDelivered", "sum"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # Build continuous hourly index
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

    hourly["session_count"] = hourly["session_count"].astype(int)
    hourly["total_kwh"] = hourly["total_kwh"].astype(float)

    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["day_of_week"] = hourly["hour"].dt.dayofweek
    hourly["month"] = hourly["hour"].dt.month
    hourly["is_weekend"] = hourly["day_of_week"].isin([5, 6]).astype(int)

    return hourly


def main() -> None:
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