import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional
import seaborn as sns
import numpy as np
import pandas as pd


class TechnicalVisualizer:
    """Enhanced visualization for all technical indicators"""

    def __init__(self):
        plt.style.use('seaborn-v0_8')  # or try 'seaborn-darkgrid', etc.
        sns.set_palette("husl")

    def plot_indicators(self, df: pd.DataFrame, ticker: str,
                        indicator_groups: List[str] = ['trend', 'momentum', 'volume', 'volatility']) -> plt.Figure:
        """
        Create comprehensive technical analysis dashboard
        Args:
            df: DataFrame with calculated indicators
            ticker: Stock ticker symbol
            indicator_groups: Which indicator groups to include
        Returns:
            matplotlib Figure object
        """
        num_plots = len(indicator_groups) + 1  # +1 for price chart
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), sharex=True)
        fig.suptitle(f'Technical Analysis Dashboard - {ticker}', y=1.02)

        if num_plots == 1:
            axes = [axes]  # Ensure axes is always a list

        # Price and Volume (always shown)
        ax = axes[0]
        ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=2)

        # Plot moving averages
        for ma in [col for col in df.columns if 'SMA_' in col or 'EMA_' in col]:
            ax.plot(df.index, df[ma], label=ma, alpha=0.7)

        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add Bollinger Bands if available
        if all(col in df.columns for col in ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']):
            ax.fill_between(df.index, df['BB_LOWER'], df['BB_UPPER'],
                            color='blue', alpha=0.1, label='Bollinger Bands')
            ax.plot(df.index, df['BB_MIDDLE'], color='blue', alpha=0.5, linestyle='--')

        # Plot each indicator group
        for i, group in enumerate(indicator_groups, start=1):
            ax = axes[i]
            self._plot_indicator_group(ax, df, group)

        plt.tight_layout()
        return fig

    def _plot_indicator_group(self, ax, df, group):
        """Plot indicators for a specific group"""
        if group == 'trend':
            self._plot_trend_indicators(ax, df)
        elif group == 'momentum':
            self._plot_momentum_indicators(ax, df)
        elif group == 'volume':
            self._plot_volume_indicators(ax, df)
        elif group == 'volatility':
            self._plot_volatility_indicators(ax, df)

    def _plot_trend_indicators(self, ax, df):
        """Plot trend indicators"""
        if 'ADX' in df.columns:
            ax.plot(df.index, df['ADX'], label='ADX (14)', color='purple')
            ax.axhline(25, color='gray', linestyle='--', alpha=0.5)

        if 'AROON_UP' in df.columns and 'AROON_DOWN' in df.columns:
            ax.plot(df.index, df['AROON_UP'], label='Aroon Up', color='green')
            ax.plot(df.index, df['AROON_DOWN'], label='Aroon Down', color='red')

        if 'TRIX' in df.columns:
            ax.plot(df.index, df['TRIX'], label='TRIX', color='blue')

        ax.set_ylabel('Trend Strength')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    def _plot_momentum_indicators(self, ax, df):
        """Plot momentum indicators"""
        if 'RSI' in df.columns:
            ax.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
            ax.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(30, color='green', linestyle='--', alpha=0.5)

        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            ax.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax.plot(df.index, df['MACD_Signal'], label='Signal', color='orange')
            ax.bar(df.index, df['MACD_Hist'], label='Histogram', color='gray', alpha=0.5)

        if 'CCI' in df.columns:
            ax.plot(df.index, df['CCI'], label='CCI', color='green')
            ax.axhline(100, color='red', linestyle='--', alpha=0.5)
            ax.axhline(-100, color='green', linestyle='--', alpha=0.5)

        ax.set_ylabel('Momentum')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    def _plot_volume_indicators(self, ax, df):
        """Plot volume indicators"""
        if 'Volume' in df.columns:
            ax.bar(df.index, df['Volume'], label='Volume', color='lightblue', alpha=0.7)

        if 'OBV' in df.columns:
            ax.plot(df.index, df['OBV'], label='OBV', color='darkblue')

        if 'MFI' in df.columns:
            ax.plot(df.index, df['MFI'], label='MFI', color='purple', alpha=0.7)
            ax.axhline(80, color='red', linestyle='--', alpha=0.5)
            ax.axhline(20, color='green', linestyle='--', alpha=0.5)

        ax.set_ylabel('Volume')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    def _plot_volatility_indicators(self, ax, df):
        """Plot volatility indicators"""
        if 'ATR' in df.columns:
            ax.plot(df.index, df['ATR'], label='ATR (14)', color='red')

        if 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns:
            ax.plot(df.index, (df['Close'] - df['BB_LOWER']) /
                    (df['BB_UPPER'] - df['BB_LOWER']),
                    label='BB %B', color='blue')
            ax.axhline(0.8, color='red', linestyle='--', alpha=0.5)
            ax.axhline(0.2, color='green', linestyle='--', alpha=0.5)

        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    def plot_correlation_heatmap(self, indicator_df: pd.DataFrame) -> plt.Figure:
        """Plot correlation heatmap of technical indicators"""
        # Select only indicator columns (exclude OHLCV)
        indicator_cols = [col for col in indicator_df.columns
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

        if not indicator_cols:
            raise ValueError("No technical indicators found in DataFrame")

        corr = indicator_df[indicator_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Technical Indicator Correlation Matrix')
        plt.tight_layout()
        return fig