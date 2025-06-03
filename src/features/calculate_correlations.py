import pandas as pd

def calculate_correlation(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation between sentiment metrics and stock returns.

    Args:
        merged_data: DataFrame containing both sentiment and return data

    Returns:
        DataFrame with correlation results for each metric
    """
    sentiment_metrics = [
        'vader_mean',
        'vader_median',
        'textblob_mean',
        'positive_pct',
        'negative_pct'
    ]

    correlations = {}
    for metric in sentiment_metrics:
        if metric in merged_data.columns:
            valid_data = merged_data[['Daily_Return', metric]].dropna()

            if (
                not valid_data.empty
                and valid_data['Daily_Return'].std() != 0
                and valid_data[metric].std() != 0
            ):
                corr = valid_data['Daily_Return'].corr(valid_data[metric], method='pearson')
                correlations[metric] = corr
            else:
                correlations[metric] = float('nan')

    results = pd.DataFrame.from_dict(correlations, orient='index', columns=['Pearson_Correlation'])
    results['Absolute_Correlation'] = results['Pearson_Correlation'].abs()
    results['Strength'] = results['Absolute_Correlation'].apply(
        lambda x: 'Strong' if x > 0.5 else ('Moderate' if x > 0.3 else 'Weak')
    )

    return results.sort_values('Absolute_Correlation', ascending=False)


def calculate_lagged_correlation(merged_data: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """
    Calculate correlations with various time lags between sentiment and returns.
    """
    results = []

    for lag in range(0, max_lag + 1):
        temp_df = merged_data.copy()
        if lag > 0:
            temp_df['Daily_Return'] = temp_df['Daily_Return'].shift(-lag)

        valid_data = temp_df[['Daily_Return', 'vader_mean']].dropna()
        if (
            not valid_data.empty
            and valid_data['Daily_Return'].std() != 0
            and valid_data['vader_mean'].std() != 0
        ):
            corr = valid_data['Daily_Return'].corr(valid_data['vader_mean'], method='pearson')
        else:
            corr = float('nan')

        results.append({
            'Lag_Days': lag,
            'Pearson_Correlation': corr,
            'Absolute_Correlation': abs(corr) if pd.notna(corr) else float('nan'),
            'Direction': 'Positive' if pd.notna(corr) and corr > 0 else 'Negative'
        })

    return pd.DataFrame(results).sort_values('Absolute_Correlation', ascending=False)
