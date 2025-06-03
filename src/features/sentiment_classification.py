import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Tuple




def classify_sentiment(df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame:
    """
    Classify sentiment of text using both VADER and TextBlob.

    Args:
        df: DataFrame containing text data.
        text_column: Name of the column with text to analyze.

    Returns:
        DataFrame with added sentiment columns:
        - VADER: compound, neg, neu, pos, sentiment_class (positive/negative/neutral)
        - TextBlob: polarity, subjectivity
    """
    # Initialize analyzers
    sia = SentimentIntensityAnalyzer()

    # Make a copy to avoid modifying original
    result_df = df.copy()

    # --- VADER Sentiment ---
    result_df['vader_scores'] = result_df[text_column].apply(
        lambda x: sia.polarity_scores(str(x))
    )
    # Extract VADER scores to separate columns
    result_df['vader_compound'] = result_df['vader_scores'].apply(lambda x: x['compound'])
    result_df['vader_neg'] = result_df['vader_scores'].apply(lambda x: x['neg'])
    result_df['vader_neu'] = result_df['vader_scores'].apply(lambda x: x['neu'])
    result_df['vader_pos'] = result_df['vader_scores'].apply(lambda x: x['pos'])

    # Classify sentiment based on VADER compound score
    conditions = [
        (result_df['vader_compound'] > 0.05),
        (result_df['vader_compound'] < -0.05)
    ]
    choices = ['positive', 'negative']
    result_df['vader_sentiment'] = np.select(conditions, choices, default='neutral')

    # --- TextBlob Sentiment ---
    result_df['textblob_polarity'] = result_df[text_column].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    result_df['textblob_subjectivity'] = result_df[text_column].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity
    )

    # Optional: Add binary flags
    result_df['is_positive'] = (result_df['vader_sentiment'] == 'positive').astype(int)
    result_df['is_negative'] = (result_df['vader_sentiment'] == 'negative').astype(int)
    result_df['is_neutral'] = (result_df['vader_sentiment'] == 'neutral').astype(int)

    return result_df


def aggregate_sentiment_by_ticker_and_date(df: pd.DataFrame,
                                           date_col: str = 'clean_date',
                                           ticker_col: str = 'Ticker') -> pd.DataFrame:
    """
    Aggregate sentiment metrics by both ticker and date using classified sentiment data.

    Args:
        df: DataFrame containing classified sentiment data (from classify_sentiment)
        date_col: Name of the cleaned date column
        ticker_col: Name of the ticker column

    Returns:
        DataFrame with aggregated sentiment metrics per ticker per day
    """
    required_cols = ['vader_compound', 'vader_sentiment', 'textblob_polarity',
                     'is_positive', 'is_negative', 'is_neutral',
                     date_col, ticker_col]

    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Group by both ticker and date
    agg_df = df.groupby([ticker_col, date_col]).agg(
        vader_mean=('vader_compound', 'mean'),
        vader_median=('vader_compound', 'median'),
        textblob_mean=('textblob_polarity', 'mean'),
        total_articles=('vader_compound', 'count'),
        positive_articles=('is_positive', 'sum'),
        negative_articles=('is_negative', 'sum'),
        neutral_articles=('is_neutral', 'sum')
    ).reset_index()

    # Calculate percentages
    agg_df['positive_pct'] = agg_df['positive_articles'] / agg_df['total_articles'] * 100
    agg_df['negative_pct'] = agg_df['negative_articles'] / agg_df['total_articles'] * 100
    agg_df['neutral_pct'] = agg_df['neutral_articles'] / agg_df['total_articles'] * 100

    # Add temporal features
    agg_df['day_of_week'] = agg_df[date_col].dt.day_name()
    agg_df['week_number'] = agg_df[date_col].dt.isocalendar().week
    agg_df['month'] = agg_df[date_col].dt.month_name()

    # Sort by ticker and date for better readability
    agg_df = agg_df.sort_values([ticker_col, date_col])

    return agg_df