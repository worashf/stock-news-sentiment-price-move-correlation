import pandas as pd
from pathlib import Path
from typing import Dict, Union
import numpy as np
from datetime import datetime
import warnings


class DataLoader:
    """
    A robust stock data loader with comprehensive data validation and cleaning

    Features:
    - File existence checking
    - Column validation
    - Data type conversion
    - Missing value handling
    - Date validation
    - Price/volume validation
    - Data cleaning
    - Duplicate handling
    """

    def __init__(self, data_dir: str = '../../data/yfinance_data'):
        self.data_dir = Path(data_dir)
        self.required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        self.valid_dtypes = {
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'int64'
        }

    def _validate_file(self, ticker: str) -> Path:
        """Check if file exists and is accessible"""
        file_path = self.data_dir / f"{ticker}_historical_data.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found for {ticker} at {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path exists but is not a file: {file_path}")
        return file_path

    def _validate_columns(self, df: pd.DataFrame, ticker: str) -> None:
        """Validate required columns are present"""
        missing_cols = self.required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to proper data types"""
        for col, dtype in self.valid_dtypes.items():
            if col in df.columns:
                try:
                    if dtype == 'int64':
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    else:
                        df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to convert {col} to {dtype}: {str(e)}")
        return df

    import pandas as pd
    import warnings
    from datetime import datetime

    def _validate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize the 'Date' column in the DataFrame.

        - Converts 'Date' to datetime.
        - Removes rows with invalid or future dates.
        - Adds a normalized 'stock_date' column (date only).

        Parameters:
            df (pd.DataFrame): Input DataFrame with a 'Date' column.

        Returns:
            pd.DataFrame: Cleaned DataFrame with valid, past/present dates.
        """
        if 'Date' not in df.columns:
            return df

        try:
            # Convert to datetime, coercing errors
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['clean_date'] = df['Date']

            # Remove rows with invalid dates (NaT)
            invalid_dates = df['clean_date'].isna()
            if invalid_dates.any():
                warnings.warn(f"Removed {invalid_dates.sum()} rows with invalid dates.")
                df = df[~invalid_dates]

            # Normalize to date only (remove time component)
            df['clean_date'] = df['clean_date'].dt.date

            # Remove rows with future dates
            current_date = datetime.now().date()
            future_dates = df['clean_date'] > current_date
            if future_dates.any():
                warnings.warn(f"Removed {future_dates.sum()} rows with future dates.")
                df = df[~future_dates]

            # Convert back to datetime64[ns] for consistency
            df['Date'] = pd.to_datetime(df['Date'])
            df['clean_date'] = pd.to_datetime(df['clean_date'])

        except Exception as e:
            raise ValueError(f"Date normalization failed: {str(e)}")

        return df


    def _validate_volume(self, df: pd.DataFrame, ticker: str) -> None:
        """Validate volume column"""
        if 'Volume' in df.columns:
            # Check for negative volume
            if (df['Volume'] < 0).any():
                raise ValueError(f"Negative volume found for {ticker}")

            # Check for zero volume on trading days
            if (df['Volume'] == 0).any():
                warnings.warn(f"Zero volume entries found for {ticker}")

    def _handle_missing_values(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Handle missing values with appropriate strategy"""
        # Count missing values before handling
        missing_before = df.isnull().sum().sum()

        # Forward fill for OHLC prices (assuming markets were closed)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Set volume to 0 if missing (assuming no trading)
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)

        # Count missing values after handling
        missing_after = df.isnull().sum().sum()

        if missing_before > 0:
            print(f"Handled {missing_before - missing_after} missing values for {ticker}")

        return df

    def _clean_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Perform final data cleaning steps"""
        # Remove duplicates while keeping first occurrence
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removed {duplicates} duplicate rows for {ticker}")
            df = df.drop_duplicates()

        # Ensure proper sorting by date
        if 'Date' in df.columns:
            df = df.sort_values('Date')
            df = df.set_index('Date')

        # Validate price consistency
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            inconsistent = (
                    (df['High'] < df['Low']) |
                    (df['High'] < df['Open']) |
                    (df['High'] < df['Close']) |
                    (df['Low'] > df['Open']) |
                    (df['Low'] > df['Close'])
            ).sum()

            if inconsistent > 0:
                warnings.warn(f"{inconsistent} inconsistent price bars found for {ticker}")

        return df

    def load_single_stock(self, ticker: str) -> pd.DataFrame:
        """
        Load and validate data for a single stock.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: Cleaned and validated DataFrame.
        """
        file_path = self._validate_file(ticker)

        try:
            # Load data with error handling for malformed CSV
            df = pd.read_csv(file_path, na_values=['', 'NA', 'N/A', 'NaN', 'null'])

            # Validate and clean data
            self._validate_columns(df, ticker)
            df = self._convert_dtypes(df)
            df = self._validate_dates(df)

            self._validate_volume(df, ticker)
            df = self._handle_missing_values(df, ticker)
            df = self._clean_data(df, ticker)

            print(f"Successfully loaded and cleaned {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            raise ValueError(f"Error processing {ticker} data: {str(e)}")

    def load_multiple_stocks(self, tickers: list) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple stocks
        Args:
            tickers: List of stock ticker symbols
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        stock_data = {}
        failed_tickers = []

        for ticker in tickers:
            try:
                stock_data[ticker] = self.load_single_stock(ticker)
            except Exception as e:
                print(f"Failed to load {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        if failed_tickers:
            print(f"\nFailed to load {len(failed_tickers)}/{len(tickers)} tickers: {failed_tickers}")

        return stock_data



# Example Usage
if __name__ == "__main__":
    try:
        loader = DataLoader()

        # Load multiple stocks
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META']
        stock_data = loader.load_multiple_stocks(tickers)

        # Display sample of loaded data
        if 'AAPL' in stock_data:
            print("\nSample AAPL data:")
            print(stock_data['AAPL'].head(3))
            print("\nData info:")
            print(stock_data['AAPL'].info())

            # Check for missing values
            print("\nMissing values check:")
            print(stock_data['AAPL'].isnull().sum())

    except Exception as e:
        print(f"Data loading failed: {str(e)}")