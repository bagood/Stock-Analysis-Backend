import numpy as np
import pandas as pd

# These are assumed to be helper functions from your project structure.
from technicalIndicators.helpersFunctions.main import retrieve_linreg_gradients, group_trends, scale_indicators

def _calculate_money_flow_multiplier(data):
    """
    (Internal Helper) Calculates the Money Flow Multiplier (MFM).

    The MFM determines the proportion of the volume that should be considered
    accumulating or distributing based on where the price closes within its
    daily range.

    Formula:
    $$ MFM = \frac{(\text{Close} - \text{Low}) - (\text{High} - \text{Close})}{\text{High} - \text{Low}} $$

    Args:
        data (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series of the Money Flow Multiplier values.
    """
    # Calculate the MFM. A check for (High - Low) being zero is implicitly handled
    # by pandas, which will result in inf or NaN, which should be handled later.
    return ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])

def _calculate_money_flow_volume(data):
    """
    (Internal Helper) Calculates the Money Flow Volume (MFV).

    MFV weights the daily volume by the Money Flow Multiplier to get a
    volume value that reflects buying or selling pressure.

    Args:
        data (pd.DataFrame): DataFrame with 'MFM' and 'Volume' columns.

    Returns:
        pd.Series: A pandas Series of the Money Flow Volume values.
    """
    # Multiply the Money Flow Multiplier by the period's volume.
    return data['MFM'] * data['Volume']

def _identfy_accumulation_distribution_indicator(data):
    """
    (Internal Helper) Calculates a custom, non-standard version of the Accumulation/Distribution (A/D) line.

    Note: The standard A/D line is a simple cumulative sum of the Money Flow Volume.
    This implementation uses a custom logic that resets the calculation every 50 periods.

    Args:
        data (pd.DataFrame): DataFrame with 'MFM' and 'MFV' columns.

    Returns:
        list: A list containing the calculated A/D indicator values.
    """
    # Initialize an empty list to store the A/D indicator values.
    all_ad_indicators = []
    
    # Iterate through each row of the DataFrame to calculate the A/D value.
    for i, row in data.iterrows():
        # This custom logic resets the A/D calculation every 50 periods.
        if i % 50 == 0:
            ad_indicator = row['MFM'] + row['MFV']
        # For other periods, it builds upon the previous day's value.
        else:
            ad_indicator = all_ad_indicators[i-1] + row['MFV']
            
        all_ad_indicators.append(ad_indicator)

    return all_ad_indicators

def _identify_historical_trends(data, column, rolling_window, threshold=0.1):
    """
    (Internal Helper) Identifies the historical trend of a data column using linear regression.

    This function calculates the slope of the data over a rolling window and classifies
    it as 'up', 'down', or 'sideways' based on the slope's value relative to a threshold.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to analyze.
        rolling_window (int): The number of periods to include in the trend calculation.
        threshold (float): The slope value threshold for classifying a trend.

    Returns:
        np.array: An array of the classified trends for each period.
    """
    # For each day 'i', calculate the linear regression gradient of the preceding 'rolling_window' days.
    # The 'group_trends' helper function then classifies the resulting slope.
    return np.array([np.nan if i < rolling_window else group_trends(retrieve_linreg_gradients(data[column].values[i-rolling_window:i]), threshold) for i in range(len(data))])

def identify_ad_indicators(data, rolling_window, threshold=0.1):
    """
    Calculates the Accumulation/Distribution line and identifies divergences
    between its trend and the price trend.

    This function orchestrates the process of calculating the A/D line, scaling it,
    and then comparing its historical trend to the historical trend of the closing price
    to find potential bullish or bearish divergences.

    Args:
        data (pd.DataFrame): The input DataFrame.
        rolling_window (int): The period for calculating historical trends.
        threshold (float): The threshold for trend classification.

    Returns:
        pd.DataFrame: The original DataFrame with A/D and price trend indicators added.
    """
    # Create a copy of the data to avoid modifying the original DataFrame.
    temp_data = data.copy()
    
    # Step 1: Calculate Money Flow Multiplier and Money Flow Volume.
    temp_data['MFM'] = _calculate_money_flow_multiplier(temp_data)
    temp_data['MFV'] = _calculate_money_flow_volume(temp_data)
    
    # Step 2: Calculate the custom A/D indicator and scale its values.
    temp_data['AD'] = _identfy_accumulation_distribution_indicator(temp_data)
    temp_data['AD'] = scale_indicators(temp_data, 'AD')

    # Step 3: Identify historical trends for both the A/D line and the closing price.
    temp_data[f'{rolling_window} Days Historical AD Trend'] = _identify_historical_trends(temp_data, 'AD', rolling_window, threshold)
    temp_data[f'{rolling_window} Days Historical Trend'] = _identify_historical_trends(temp_data, 'Close', rolling_window, threshold)

    # Step 4: Classify the A/D trend into separate Uptrend, Downtrend, and Sideways columns.
    data[f'{rolling_window} Days Historical AD Uptrend'] = temp_data[f'{rolling_window} Days Historical AD Trend'].apply(lambda row: row >= 0.2).astype(int)
    data[f'{rolling_window} Days Historical AD Downtrend'] = temp_data[f'{rolling_window} Days Historical AD Trend'].apply(lambda row: row <= 0.2).astype(int)
    data[f'{rolling_window} Days Historical AD Sideways'] = temp_data[f'{rolling_window} Days Historical AD Trend'].apply(lambda row: np.abs(row) < 0.2).astype(int)

    # Step 5: Classify the price trend into separate Uptrend, Downtrend, and Sideways columns.
    data[f'{rolling_window} Days Historical Uptrend'] = temp_data[f'{rolling_window} Days Historical Trend'].apply(lambda row: row >= 0.2).astype(int)
    data[f'{rolling_window} Days Historical Downtrend'] = temp_data[f'{rolling_window} Days Historical Trend'].apply(lambda row: row <= 0.2).astype(int)
    data[f'{rolling_window} Days Historical Sideways'] = temp_data[f'{rolling_window} Days Historical Trend'].apply(lambda row: np.abs(row) < 0.2).astype(int)
    
    return data
