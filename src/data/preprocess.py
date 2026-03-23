import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase and replace spaces with underscores.
    """
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates().copy()


def parse_datetime_columns(df: pd.DataFrame, datetime_columns: list[str]) -> pd.DataFrame:
    """
    Convert selected columns to datetime.
    """
    df = df.copy()
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def basic_cleaning(df: pd.DataFrame, datetime_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Apply standard cleaning steps.
    """
    df = standardize_column_names(df)
    df = remove_duplicates(df)

    if datetime_columns:
        df = parse_datetime_columns(df, datetime_columns)

    return df