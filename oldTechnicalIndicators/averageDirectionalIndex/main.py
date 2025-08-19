import numpy as np
import pandas as pd

# These are assumed to be helper functions from your project structure.
from oldTechnicalIndicators.helpersFunctions.main import backfill_inf_values, scale_indicators

def _calculate_average_true_range(row):
    """
    (Internal Helper) Calculates the True Range (TR) for a single data row.

    The True Range is a measure of daily volatility. This implementation defines it as the
    greatest of the following three values:
    1. The current day's high minus the current day's low.
    2. The current day's high minus the current day's close.
    3. The current day's close minus the current day's low.
    
    Formula:
    $$ TR = \max(\text{High} - \text{Low}, \text{High} - \text{Close}, \text{Close} - \text{Low}) $$

    Args:
        row (pd.Series): A row from the price DataFrame.

    Returns:
        float: The calculated True Range value for that row.
    """
    # Return the maximum of the three range calculations for the given row.
    return np.max([row['High']-row['Low'], row['High']-row['Close'], row['Close']-row['Low']])

def _calculate_moving_average(data, column, rolling_window=14):
    """
    (Internal Helper) Calculates the Simple Moving Average (SMA) for a given column.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to calculate the SMA on.
        rolling_window (int): The number of periods for the moving average.

    Returns:
        pd.Series: A pandas Series containing the SMA values.
    """
    # Use pandas' built-in rolling mean function.
    return data[column].rolling(rolling_window).mean()

def _identify_directional_movements(data, movements):
    """
    (Internal Helper) Calculates the raw directional movement (+DM or -DM).

    - Positive Directional Movement (+DM): Current High - Previous High
    - Negative Directional Movement (-DM): Previous Low - Current Low

    Args:
        data (pd.DataFrame): The input DataFrame with 'High' and 'Low' prices.
        movements (str): '+' for positive DM or '-' for negative DM.

    Returns:
        np.array: An array of the calculated raw directional movements.
    """
    # Calculate the difference between the current high and the previous high.
    if movements == '+':
        return np.concatenate(([np.nan], data['High'].values[1:]-data['High'].values[:-1]))
    
    # Calculate the difference between the previous low and the current low.
    elif movements == '-':
        return np.concatenate(([np.nan], data['Low'].values[:-1]-data['Low'].values[1:]))

def _identify_directional_index(data, rolling_window=14):
    """
    (Internal Helper) Calculates the Directional Indicators (+DI, -DI) and the base Directional Index (DI/DX).
    
    The +DI and -DI measure the strength of the upward and downward trends, respectively. The DI (or DX)
    measures the absolute difference between them to quantify overall trend strength.
    
    Formula for Directional Index (DX):
    $$ DX = \frac{|\text{+DI} - \text{-DI}|}{|\text{+DI} + \text{-DI}|} \times 100 $$

    Args:
        data (pd.DataFrame): DataFrame containing SMA, ATR, +DM, and -DM values.
        rolling_window (int): The period used for the calculations.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple containing +DI, -DI, and DI.
    """
    # Calculate the Positive Directional Indicator (+DI).
    positive_di = 100 * data[f'{rolling_window} Days SMA'] * data['+DM'] / data[f'{rolling_window} Days ATR']
    # Calculate the Negative Directional Indicator (-DI).
    negative_di = 100 * data[f'{rolling_window} Days SMA'] * data['-DM'] / data[f'{rolling_window} Days ATR']
    # Calculate the base Directional Index (DX) from +DI and -DI.
    di = np.abs(positive_di - negative_di) / np.abs(positive_di + negative_di) * 100
    
    return (positive_di, negative_di, di)

def _calculate_average_directional_index(data, rolling_window=14):
    """
    (Internal Helper) Calculates the Average Directional Index (ADX).
    
    The ADX is a smoothed moving average of the Directional Index (DI/DX) and is used
    to measure the strength of a trend, not its direction.
    
    Formula (Wilder's Smoothing):
    $$ ADX_{\text{current}} = \frac{(ADX_{\text{previous}} \times (n-1)) + DX_{\text{current}}}{n} $$

    Args:
        data (pd.DataFrame): DataFrame containing the base DI values.
        rolling_window (int): The smoothing period (n).

    Returns:
        np.array: An array of the calculated ADX values.
    """
    # Apply Wilder's smoothing average to the base DI values to get the ADX.
    return np.concatenate(([np.nan], ((data[f'DI {rolling_window}'].values[:-1] * (rolling_window-1)) + data[f'DI {rolling_window}'].values[1:]) / rolling_window))

def _identify_adx_indicators(data, rolling_window=14):
    """
    (Internal Helper) Generates trading signals based on ADX and DI values.

    - Uptrend: ADX is above the threshold (strong trend) and +DI is greater than -DI.
    - Downtrend: ADX is above the threshold (strong trend) and +DI is less than -DI.
    - Weak Trend: ADX is below the threshold.

    Args:
        data (pd.DataFrame): DataFrame with ADX, +DI, and -DI columns.
        rolling_window (int): The period used for the indicator names.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple with uptrend, downtrend, and weak trend signals.
    """
    # Define column names for easier access.
    di_plus_col = f'+DI {rolling_window}'
    di_minus_col = f'-DI {rolling_window}'
    adx_col = f'ADX {rolling_window}'
    
    # Identify a strong uptrend: ADX >= 0.2 AND +DI > -DI.
    adx_uptrend = data.apply(lambda row: (row[adx_col] >= 0.2) * (row[di_plus_col] > row[di_minus_col]) , axis=1)
    # Identify a strong downtrend: ADX >= 0.2 AND +DI < -DI.
    adx_downtrend = data.apply(lambda row: (row[adx_col] >= 0.2) * (row[di_plus_col] < row[di_minus_col]) , axis=1)
    # Identify a weak or non-trending market: ADX < 0.2.
    adx_weaktrend = data.apply(lambda row: row[adx_col] < 0.2, axis=1).astype(int)

    return (adx_uptrend, adx_downtrend, adx_weaktrend)

def identify_adx_indicators(data, rolling_window=14):
    """
    Calculates the Average Directional Index (ADX) and its related trading signals.

    This function orchestrates the entire ADX calculation pipeline, from raw price
    data to final trend strength signals, and merges the results back into the
    original DataFrame.

    Args:
        data (pd.DataFrame): DataFrame with 'Date', 'High', 'Low', and 'Close' columns.
        rolling_window (int): The period to use for all calculations (default is 14).

    Returns:
        pd.DataFrame: The original DataFrame with added columns for ADX signals.
    """
    # Create a copy to avoid modifying the original DataFrame.
    temp_data = data.copy()

    # Step 1: Calculate True Range (TR), and its moving average (ATR), plus a price SMA.
    temp_data['TR'] = temp_data.apply(lambda row: _calculate_average_true_range(row), axis=1)
    temp_data[f'{rolling_window} Days SMA'] = _calculate_moving_average(temp_data, 'Close', rolling_window)
    temp_data[f'{rolling_window} Days ATR'] = _calculate_moving_average(temp_data, 'TR', rolling_window)
    
    # Step 2: Calculate Positive and Negative Directional Movements (+DM, -DM).
    temp_data['+DM'] = _identify_directional_movements(temp_data, '+')
    temp_data['-DM'] = _identify_directional_movements(temp_data, '-')
    
    # Step 3: Calculate Directional Indicators (+DI, -DI) and the base DI (DX).
    directions = _identify_directional_index(temp_data, rolling_window)
    temp_data[f'+DI {rolling_window}'] = directions[0]
    temp_data[f'-DI {rolling_window}'] = directions[1]
    temp_data[f'DI {rolling_window}'] = directions[2]

    # Step 4: Calculate the final Average Directional Index (ADX).
    temp_data[f'ADX {rolling_window}'] = _calculate_average_directional_index(temp_data, rolling_window)

    # Step 5: Generate trend signals from the calculated ADX and DI values.
    adx_indicators = _identify_adx_indicators(temp_data, rolling_window)
    temp_data[f'ADX Uptrend {rolling_window}'] = adx_indicators[0]
    temp_data[f'ADX Downtrend {rolling_window}'] = adx_indicators[1]
    temp_data[f'ADX Weaktrend {rolling_window}'] = adx_indicators[2]

    # Step 6: Post-process the data (e.g., handle infinite values and scale the final ADX).
    temp_data = backfill_inf_values(temp_data, f'ADX {rolling_window}')
    temp_data[f'ADX {rolling_window}'] = scale_indicators(temp_data, f'ADX {rolling_window}')

    # Step 7: Clean up rows with initial NaN values and select columns to merge.
    temp_data.dropna(inplace=True)
    columns_to_merge = [
        'Date',
        f'ADX Uptrend {rolling_window}',
        f'ADX Downtrend {rolling_window}',
        f'ADX Weaktrend {rolling_window}'
    ]
    # Merge the final signals back into the original DataFrame.
    data = pd.merge(data, temp_data[columns_to_merge], on='Date', how='left')

    return data