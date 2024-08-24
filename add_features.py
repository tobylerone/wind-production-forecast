import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame, num_fourier_terms: int = 2) -> pd.DataFrame:
    """
    Adds discrete calendar features (ordinal step functions) and continuous Fourier terms to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        num_fourier_terms (int): Number of Fourier terms to add (default is 5).
    
    Returns:
        pd.DataFrame: DataFrame with added features.

    Example:
        If num_fourier_terms = 2, this will create sine and cosine wave features with
        wavelengths corresponding to one full cycle per day (i=1), and two full cycles
        per day (i=2).
    """
    
    # Add discrete calendar features (Essentially an ordinal step function)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Week'] = df.index.isocalendar().week
    df['DayOfWeek'] = df.index.dayofweek
    df['Hour'] = df.index.hour

    # Add continuous calendar features (Continuous fourier terms)
    for i in range(1, num_fourier_terms + 1):
        freq = 2 * np.pi * i / 24  # Daily frequency (24 hours)
        df[f'Sin_{i}'] = np.sin(freq * df.index.hour)
        df[f'Cos_{i}'] = np.cos(freq * df.index.hour)

    return df

def add_time_lags(series: pd.Series, lags: int = 48) -> pd.DataFrame:

    df = pd.DataFrame()
    
    for i in range(1, lags + 1):
        df[f"Lagged_Potential_{i}h"] = series.shift(i)
        
    # Add target back
    df['Wind_Energy_Potential'] = series

    return df

def add_aggregations(df: pd.DataFrame) -> pd.DataFrame:

    # Add rolling window aggregations

    # Add seasonal rolling window aggregations

    # Add exponentially weighted moving averages

    #
    pass