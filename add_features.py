import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame, datetime_feature: str, num_fourier_terms: int = 2) -> pd.DataFrame:
    """
    Adds discrete calendar features (ordinal step functions) and continuous Fourier terms to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        datetime_feature: Feature to use for time series information. Defaults to index
        num_fourier_terms (int): Number of Fourier terms to add (default is 5).
    
    Returns:
        pd.DataFrame: DataFrame with added features.

    Example:
        If num_fourier_terms = 2, this will create sine and cosine wave features with
        wavelengths corresponding to one full cycle per day (i=1), and two full cycles
        per day (i=2).
    """

    datetime_series = df[datetime_feature].dt if datetime_feature else df.index
    
    # Add discrete calendar features (Essentially an ordinal step function)
    df['Year'] = datetime_series.year
    df['Month'] = datetime_series.month
    df['Week'] = datetime_series.isocalendar().week
    df['DayOfWeek'] = datetime_series.dayofweek
    df['Hour'] = datetime_series.hour

    # Add continuous calendar features (Continuous fourier terms)
    for i in range(1, num_fourier_terms + 1):
        freq = 2 * np.pi * i / 24  # Daily frequency (24 hours)
        df[f'Sin_{i}'] = np.sin(freq * datetime_series.hour)
        df[f'Cos_{i}'] = np.cos(freq * datetime_series.hour)

    return df

def add_time_lags(df: pd.DataFrame, target_key: str, lags: int = 48) -> pd.DataFrame:

    for i in range(1, lags + 1):
        df[f"Lagged_Potential_{i}h"] = df[target_key].shift(i)

    return df

def _add_ewma_features(df: pd.DataFrame, target_key: str, max_span: int = 3) -> pd.DataFrame:

    # Shift to avoid leaking lag=0 into new features
    for span in range(2, max_span):
        df[f'{target_key}_EWMA_SPAN_{span}'] = df[target_key].ewm(span=span).mean().shift(1)

    return df

def add_aggregations(df: pd.DataFrame) -> pd.DataFrame:

    # Add rolling window aggregations

    # Add seasonal rolling window aggregations

    # Add exponentially weighted moving averages
    df = _add_ewma_features(df, 'Wind_Energy_Potential', 14)

    return df