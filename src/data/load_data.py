import pandas as pd
from pathlib import Path


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)
    return df


def preview_data(df: pd.DataFrame, n: int = 5) -> None:
    """
    Print a quick preview of the dataset.
    """
    print("\nFirst rows:")
    print(df.head(n))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())