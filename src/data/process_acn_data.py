from __future__ import annotations

# Standard library imports
import json
from pathlib import Path

# Third-party imports
import pandas as pd


# Input raw JSON file from the ACN API
RAW_FILE = Path("data/raw/acn_caltech_sessions_2019_2021.json")

# Output folder for processed data
PROCESSED_DIR = Path("data/processed")


def load_raw_json(file_path: Path) -> list[dict]:
    """
    Load the raw ACN JSON file from disk.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    return sessions


def flatten_sessions(sessions: list[dict]) -> pd.DataFrame:
    """
    Flatten the raw JSON list into a pandas DataFrame.
    Each row represents one EV charging session.
    """
    df = pd.json_normalize(sessions)
    return df


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the flattened session table.

    Main operations:
    - convert time columns to datetime
    - keep the most relevant variables
    - remove rows with missing essential fields
    - sort sessions chronologically
    """
    df = df.copy()

    # Convert important time columns into datetime format
    datetime_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Keep only the main columns needed for analysis and forecasting
    desired_cols = [
        "_id",
        "connectionTime",
        "disconnectTime",
        "doneChargingTime",
        "kWhDelivered",
        "siteID",
        "stationID",
        "spaceID",
        "timezone",
        "userInputs",
    ]
    existing_cols = [col for col in desired_cols if col in df.columns]
    df = df[existing_cols].copy()

    # Remove rows missing essential forecasting information
    if "connectionTime" in df.columns:
        df = df[df["connectionTime"].notna()]
    if "kWhDelivered" in df.columns:
        df = df[df["kWhDelivered"].notna()]

    # Sort by session start time
    if "connectionTime" in df.columns:
        df = df.sort_values("connectionTime").reset_index(drop=True)

    return df


def save_processed_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the cleaned session-level dataset as a CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed session data to: {output_path}")
    print(f"Shape: {df.shape}")


def main() -> None:
    """
    Main preprocessing step:
    - load raw JSON
    - flatten into a table
    - clean and keep useful columns
    - save the processed session-level CSV
    """
    sessions = load_raw_json(RAW_FILE)
    print(f"Loaded {len(sessions)} raw sessions.")

    df = flatten_sessions(sessions)
    print(f"Flattened shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    df_clean = clean_sessions(df)
    save_processed_csv(df_clean, PROCESSED_DIR / "acn_sessions_flat.csv")


if __name__ == "__main__":
    main()