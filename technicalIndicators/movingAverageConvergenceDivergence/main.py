import numpy as np
import pandas as pd

def _calculate_moving_average_convergence_divergence(data):
    """
    (Internal Helper) Calculates the Moving Average Convergence Divergence (MACD) line.

    The MACD line is the core component of the indicator and is calculated by
    subtracting the 26-period Exponential Moving Average (EMA) from the
    12-period EMA.

    Formula:
    $$ \text{MACD Line} = \text{12-Period EMA} - \text{26-Period EMA} $$

    Args:
        data (pd.DataFrame): DataFrame containing a 'Close' price column.

    Returns:
        pd.Series: A pandas Series of the calculated MACD line values.
    """
    # Calculate the 12-period and 26-period Exponential Moving Averages (EMAs) of the closing price.
    ema_short = data['Close'].ewm(span=12).mean()
    ema_long = data['Close'].ewm(span=26).mean()
    
    # Subtract the long-period EMA from the short-period EMA to get the MACD line.
    return ema_short - ema_long

def identify_macd_indicators(data):
    """
    Calculates the MACD line and generates simple trend signals based on its value.

    This function calculates the MACD line and then classifies the trend based on
    whether the MACD value is strongly positive (>= 0.5), strongly negative (<= -0.5),
    or neutral.

    Note: This is a custom interpretation and does not calculate the standard MACD
    Signal Line or Histogram, which are typically used for crossover signals.

    Args:
        data (pd.DataFrame): The input DataFrame with a 'Close' price column.

    Returns:
        pd.DataFrame: The original DataFrame with 'MACD Upwards', 'MACD Downwards',
                      and 'MACD Sideways' signal columns added.
    """
    # Create a copy of the data to avoid modifying the original DataFrame.
    temp_data = data.copy()

    # Step 1: Calculate the MACD line.
    temp_data['MACD'] = _calculate_moving_average_convergence_divergence(data)
    
    # Step 2: Generate signals based on the MACD line's value relative to fixed thresholds.
    # An 'Upwards' signal is generated if the MACD is 0.5 or greater.
    data['MACD Upwards'] = temp_data['MACD'].apply(lambda row: row >= 0.5).astype(int)
    # A 'Downwards' signal is generated if the MACD is -0.5 or less.
    data['MACD Downwards'] = temp_data['MACD'].apply(lambda row: row <= -0.5).astype(int)
    # A 'Sideways' signal is generated if the MACD is between -0.5 and 0.5.
    data['MACD Sideways'] = temp_data['MACD'].apply(lambda row: np.abs(row) < 0.5).astype(int)

    return data
