from typing import Dict, List, Optional
import pandas as pd
from src import TechnicalAnalyzer
from src import FinancialMetrics
from src import TechnicalVisualizer


class TechnicalAnalysisPipeline:
    """Complete technical analysis pipeline with all indicators"""

    def __init__(self):
        self.ta = TechnicalAnalyzer()
        self.viz = TechnicalVisualizer()
        self.fin = FinancialMetrics()

    def analyze_stock(self, df: pd.DataFrame, ticker: str,
                      indicator_groups: List[str] = ['trend', 'momentum', 'volume', 'volatility']) -> Dict:
        """
        Complete technical analysis for a single stock
        Args:
            df: DataFrame with OHLCV datafin
            ticker: Stock ticker symbol
            indicator_groups: Which indicator groups to include
        Returns:
            Dictionary containing:
            - data: DataFrame with all indicators
            - metrics: Financial metrics
            - figure: Visualization figure
        """
        try:
            # Calculate all technical indicators
            df = self.ta.calculate_all_indicators(df)

            # Calculate financial metrics
            df, metrics = self.fin.calculate_all_metrics(df)

            # Create visualization
            fig = self.viz.plot_indicators(df, ticker, indicator_groups)

            return {
                'data': df,
                'metrics': metrics,
                'figure': fig
            }
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            return None

    def analyze_multiple_stocks(self, stock_data: Dict[str, pd.DataFrame],
                                indicator_groups: List[str] = ['trend', 'momentum']) -> Dict:
        """
        Analyze multiple stocks with comparative metrics
        Args:
            stock_data: Dictionary of {ticker: DataFrame}
            indicator_groups: Which indicator groups to include in visualization
        Returns:
            Dictionary containing analysis results for each stock plus correlation matrix
        """
        results = {}

        for ticker, df in stock_data.items():
            print(f"Analyzing {ticker}...")
            results[ticker] = self.analyze_stock(df, ticker, indicator_groups)

        # Add correlation matrix if we have multiple stocks
        valid_data = {t: r['data'] for t, r in results.items() if r is not None}
        if len(valid_data) > 1:
            print("Generating correlation matrix...")
            corr_fig = self.viz.plot_correlation_heatmap(
                pd.concat({t: df for t, df in valid_data.items()},
                          names=['Ticker']).reset_index(level=0)
            )
            results['correlation'] = corr_fig

        return results

    def analyze_custom_indicators(self, df: pd.DataFrame, ticker: str,
                                  indicator_list: List[str]) -> Dict:
        """
        Analyze with only specific indicators
        Args:
            df: Input DataFrame
            ticker: Stock ticker
            indicator_list: List of indicator names to calculate
        Returns:
            Analysis results dictionary
        """
        try:
            # Calculate only selected indicators
            df = self.ta.calculate_selected_indicators(df, indicator_list)

            # Create visualization focusing only on these indicators
            fig = self.viz.plot_indicators(df, ticker, [])

            return {
                'data': df,
                'figure': fig
            }
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            return None