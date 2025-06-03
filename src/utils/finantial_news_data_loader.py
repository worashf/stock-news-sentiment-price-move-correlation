import os
import pandas as pd
import warnings
from datetime import datetime
from typing import List
from typing import Optional
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def load_csv_finantial_news_data(file_path: str) -> pd.DataFrame:
    """
    Load financial news data from a CSV file, clean the date, and standardize tickers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    try:
        df = pd.read_csv(file_path, engine='python')
        print("Data loaded successfully.")
        print(f"DataFrame shape: {df.shape}")


        # Standardize ticker column
        if 'stock' in df.columns:
            df['Ticker'] = df['stock'].str.upper()
        else:
            warnings.warn("Column 'stock' not found; skipping Ticker generation.")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load and process news data: {e}")






def clean_news_dates(df: pd.DataFrame,
                     date_col: str = 'date',
                     new_col: Optional[str] = None) -> pd.DataFrame:
    """
    Robust date cleaning with proper timezone handling and comparison safety.

    Args:
        df: Input DataFrame
        date_col: Source date column name
        new_col: Optional output column name (defaults to overwriting date_col)

    Returns:
        DataFrame with cleaned datetime64[ns] dates (timezone-naive)
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found")

    df = df.copy()
    output_col = new_col if new_col else date_col

    # Convert to timezone-naive datetime64[ns]
    df[output_col] = pd.to_datetime(
        df[date_col],
        errors='coerce',
        utc=False
    ).dt.tz_localize(None)

    # Remove invalid dates
    invalid_mask = df[output_col].isna()
    if invalid_mask.any():

        warnings.warn(f"Removed {invalid_mask.sum()} invalid dates", UserWarning)
        df = df[~invalid_mask].copy()

    # Ensure comparison-safe current date
    current_date = pd.Timestamp(datetime.now().date()).tz_localize(None)

    # Remove future dates (safe comparison)
    future_mask = df[output_col] > current_date
    if future_mask.any():
        warnings.warn(f"Removed {future_mask.sum()} future dates", UserWarning)
        df = df[~future_mask].copy()

    return df

def filter_news_by_ticker(df: pd.DataFrame, tickers: List[str], ticker_col: str = 'Ticker') -> pd.DataFrame:
    """
    Filter news DataFrame by a list of stock tickers.
    """
    if ticker_col not in df.columns:
        raise KeyError(f"Column '{ticker_col}' not found in DataFrame")
    return df[df[ticker_col].isin(tickers)].copy()
