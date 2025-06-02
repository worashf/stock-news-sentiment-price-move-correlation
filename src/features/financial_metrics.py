import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings


class FinancialMetrics:
    """Financial metrics calculation with PyNance fallback to manual calculations"""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize financial metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self._has_pynance = self._check_pynance_availability()

    def _check_pynance_availability(self) -> bool:
        """Check if PyNance is properly installed and available"""
        try:
            import pynance as pn
            # Test basic functionality
            if hasattr(pn, 'returns') and hasattr(pn.returns, 'daily'):
                return True
            return False
        except ImportError:
            return False

    def _pynance_daily_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns using PyNance if available"""
        if self._has_pynance:
            import pynance as pn
            return pn.returns.daily(prices)
        return prices.pct_change()

    def _pynance_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns using PyNance if available"""
        if self._has_pynance:
            import pynance as pn
            return pn.returns.log(prices)
        return np.log(prices / prices.shift(1))

    def _pynance_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio using PyNance if available"""
        if self._has_pynance:
            import pynance as pn
            return pn.stats.sharpe_ratio(returns, risk_free=self.risk_free_rate)

        # Manual calculation
        excess_returns = returns - (self.risk_free_rate / 252)
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _pynance_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate max drawdown using PyNance if available"""
        if self._has_pynance:
            import pynance as pn
            return pn.stats.max_drawdown(prices)

        # Manual calculation
        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max
        return drawdown.min()

    def _pynance_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using PyNance if available"""
        if self._has_pynance:
            import pynance as pn
            return pn.stats.sortino_ratio(returns, risk_free=self.risk_free_rate)

        # Manual calculation
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std

    def calculate_returns(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """Calculate return metrics with robust error handling

        Args:
            df: DataFrame with price data
            price_col: Name of column containing price data

        Returns:
            DataFrame with additional return columns
        """
        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")

        result = df.copy()

        try:
            result['Daily_Return'] = self._pynance_daily_returns(result[price_col])
            result['Log_Return'] = self._pynance_log_returns(result[price_col])
            result['Cumulative_Return'] = (1 + result['Daily_Return']).cumprod() - 1
        except Exception as e:
            warnings.warn(f"Error calculating returns: {str(e)}. Using simple pct_change()")
            result['Daily_Return'] = result[price_col].pct_change()
            result['Log_Return'] = np.log(result[price_col] / result[price_col].shift(1))
            result['Cumulative_Return'] = (1 + result['Daily_Return']).cumprod() - 1

        return result

    def calculate_risk_metrics(self, df: pd.DataFrame, returns_col: str = 'Daily_Return') -> Dict[str, float]:
        """Calculate key risk metrics with robust error handling

        Args:
            df: DataFrame with return data
            returns_col: Name of column containing return data

        Returns:
            Dictionary of risk metrics
        """
        if returns_col not in df.columns:
            raise ValueError(f"Returns column '{returns_col}' not found in DataFrame")

        metrics = {}
        returns = df[returns_col].dropna()

        try:
            metrics['Sharpe_Ratio'] = self._pynance_sharpe_ratio(returns)
        except Exception as e:
            warnings.warn(f"Error calculating Sharpe ratio: {str(e)}")
            metrics['Sharpe_Ratio'] = np.nan

        try:
            metrics['Max_Drawdown'] = self._pynance_max_drawdown(df['Close'])
        except Exception as e:
            warnings.warn(f"Error calculating max drawdown: {str(e)}")
            metrics['Max_Drawdown'] = np.nan

        try:
            metrics['Annualized_Volatility'] = returns.std() * np.sqrt(252)
        except Exception as e:
            warnings.warn(f"Error calculating volatility: {str(e)}")
            metrics['Annualized_Volatility'] = np.nan

        try:
            metrics['Sortino_Ratio'] = self._pynance_sortino_ratio(returns)
        except Exception as e:
            warnings.warn(f"Error calculating Sortino ratio: {str(e)}")
            metrics['Sortino_Ratio'] = np.nan

        return metrics

    def calculate_all_metrics(self, df: pd.DataFrame,
                              price_col: str = 'Close',
                              returns_col: str = 'Daily_Return') -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate all financial metrics with comprehensive error handling

        Args:
            df: Input DataFrame
            price_col: Column name for price data
            returns_col: Column name to use for returns (will be created if not exists)

        Returns:
            Tuple of (DataFrame with metrics, Dictionary of risk metrics)
        """
        try:
            # Calculate returns if not already present
            if returns_col not in df.columns:
                df = self.calculate_returns(df, price_col)

            # Calculate risk metrics
            metrics = self.calculate_risk_metrics(df, returns_col)

            return df, metrics

        except Exception as e:
            warnings.warn(f"Error in calculate_all_metrics: {str(e)}")
            # Return original DataFrame and empty metrics on error
            return df, {}