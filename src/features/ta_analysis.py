import talib
import pandas as pd
from typing import Dict, List, Optional
import warnings


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis using TA-Lib with all requested indicators
    organized into logical categories.
    """

    # Indicator categories
    TREND_INDICATORS = [
        'ADX', 'ADXR', 'AROON', 'AROONOSC', 'DX',
        'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'TRIX'
    ]

    MOMENTUM_INDICATORS = [
        'APO', 'CCI', 'CMO', 'MACD', 'MACDEXT', 'MACDFIX',
        'MOM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100',
        'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'ULTOSC', 'WILLR'
    ]

    VOLUME_INDICATORS = [
        'MFI', 'OBV', 'BOP'
    ]

    VOLATILITY_INDICATORS = [
        'ATR', 'NATR', 'TRANGE', 'BBANDS'
    ]

    def __init__(self):
        self.available_indicators = self._get_available_indicators()

    def _get_available_indicators(self) -> List[str]:
        """Get list of all available indicators from TA-Lib"""
        return [
            'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 'CCI', 'CMO',
            'DX', 'MACD', 'MACDEXT', 'MACDFIX', 'MFI', 'MINUS_DI', 'MINUS_DM',
            'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR',
            'ROCR100', 'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX',
            'ULTOSC', 'WILLR', 'ATR', 'NATR', 'TRANGE', 'BBANDS', 'OBV'
        ]

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all trend indicators"""
        # ADX and related indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ADXR'] = talib.ADXR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['DX'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Directional Indicators
        df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
        df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)

        # Aroon indicators
        df['AROON_DOWN'], df['AROON_UP'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
        df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

        # TRIX
        df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)

        return df

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all momentum indicators"""
        # MACD family
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
        df['MACDEXT'], _, _ = talib.MACDEXT(df['Close'],
                                            fastperiod=12, fastmatype=0,
                                            slowperiod=26, slowmatype=0,
                                            signalperiod=9, signalmatype=0)
        df['MACDFIX'], _, _ = talib.MACDFIX(df['Close'], signalperiod=9)

        # Oscillators
        df['APO'] = talib.APO(df['Close'], fastperiod=12, slowperiod=26)
        df['PPO'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CMO'] = talib.CMO(df['Close'], timeperiod=14)
        df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'],
                                    timeperiod1=7, timeperiod2=14, timeperiod3=28)
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Rate of Change indicators
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        df['ROCP'] = talib.ROCP(df['Close'], timeperiod=10)
        df['ROCR'] = talib.ROCR(df['Close'], timeperiod=10)
        df['ROCR100'] = talib.ROCR100(df['Close'], timeperiod=10)
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)

        # Stochastic indicators
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                                                   fastk_period=5, slowk_period=3,
                                                   slowk_matype=0, slowd_period=3,
                                                   slowd_matype=0)
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['High'], df['Low'], df['Close'],
                                                      fastk_period=5, fastd_period=3,
                                                      fastd_matype=0)
        df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(df['Close'], timeperiod=14,
                                                            fastk_period=5, fastd_period=3,
                                                            fastd_matype=0)
        return df

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volume indicators"""
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])
        return df

    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volatility indicators"""
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])

        # Bollinger Bands
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(
            df['Close'], timeperiod=20,
            nbdevup=2, nbdevdn=2, matype=0
        )
        return df

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all available technical indicators
        Returns DataFrame with all indicators added as new columns
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            df = self.calculate_trend_indicators(df)
            df = self.calculate_momentum_indicators(df)
            df = self.calculate_volume_indicators(df)
            df = self.calculate_volatility_indicators(df)

            # Add moving averages (common baseline indicators)
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(df['Close'], timeperiod=period)

        return df

    def calculate_selected_indicators(self, df: pd.DataFrame,
                                      indicator_list: List[str]) -> pd.DataFrame:
        """
        Calculate only specified indicators
        Args:
            df: Input DataFrame with OHLCV data
            indicator_list: List of indicator names to calculate
        Returns:
            DataFrame with selected indicators added
        """
        indicator_groups = {
            'trend': self.calculate_trend_indicators,
            'momentum': self.calculate_momentum_indicators,
            'volume': self.calculate_volume_indicators,
            'volatility': self.calculate_volatility_indicators
        }

        # Calculate all indicators in each group that has at least one requested indicator
        for group_name, group_function in indicator_groups.items():
            group_indicators = getattr(self, f"{group_name.upper()}_INDICATORS")
            if any(indicator in indicator_list for indicator in group_indicators):
                df = group_function(df)

        return df