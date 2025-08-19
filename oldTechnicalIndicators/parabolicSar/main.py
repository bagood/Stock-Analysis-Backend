import numpy as np
import pandas as pd

# This script assumes the existence of a custom 'onBalanceVolume' module
# which is used to identify initial price trends.
import oldTechnicalIndicators.onBalanceVolume.main as obv

def _group_support_resistance_trend(data):
    """
    (Internal Helper) Groups consecutive days that share the same trend direction.

    This function iterates through the 'Current Trend' column and assigns a unique
    group number to each continuous block of identical trend values.

    Args:
        data (pd.DataFrame): DataFrame with a 'Current Trend' column.

    Returns:
        list: A list of group numbers corresponding to each row in the DataFrame.
    """
    # Initialize the group counter and the list to hold group numbers.
    n = 0
    group_trend = []
    
    # Iterate through the index up to the second-to-last row.
    for i in list(data.index[:-1]):
        # If the trend on the current day is different from the next day, increment the group counter.
        if data.loc[i, 'Current Trend'] != data.loc[i+1, 'Current Trend']:
            n += 1
        # Append the current group number.
        group_trend.append(n)
        
    # Append the last group number one more time for the final row.
    group_trend.append(group_trend[-1])
    
    # Ensure the generated list has the same length as the DataFrame.
    assert len(group_trend) == len(data)

    return group_trend

def _calculate_extreme_point(data):
    """
    (Internal Helper) Finds the extreme price (highest high or lowest low) for each trend group.

    For each trend group identified, this function finds the maximum price if it's an
    uptrend or the minimum price if it's a downtrend. This value serves as the
    Extreme Point (EP) in the Parabolic SAR calculation.

    Args:
        data (pd.DataFrame): DataFrame with 'Group Trend', 'Current Trend', and 'Close' columns.

    Returns:
        pd.Series: A pandas Series containing the Extreme Point for each row.
    """
    # Group the data by the 'Group Trend' and find the max and min close price within each group.
    merged_data = pd.merge(data[['Date', 'Group Trend', 'Current Trend']], data.groupby('Group Trend')['Close'].agg([np.max, np.min]).reset_index(), on='Group Trend', how='left')
    
    # For each row, select the 'max' as the Extreme Point if it's an uptrend (>0), otherwise select the 'min'.
    extreme_points =  merged_data.apply(lambda row: row['max'] if row['Current Trend'] > 0 else row['min'], axis=1)

    # Ensure the result has the same length as the original data.
    assert len(extreme_points) == len(data)
    
    return extreme_points

def _calculate_sar(data, acceleration_factor=0.02): 
    """
    (Internal Helper) Calculates the Parabolic Stop and Reverse (SAR) value.

    This function iteratively calculates the SAR for each period based on the previous
    SAR, the Extreme Point (EP), and the acceleration factor.

    Formula:
    $$ SAR_{today} = SAR_{yesterday} + AF \times (EP - SAR_{yesterday}) $$

    Args:
        data (pd.DataFrame): DataFrame with 'Extreme Point' and 'Close' columns.
        acceleration_factor (float): The acceleration factor (AF) for the SAR calculation.

    Returns:
        np.array: An array of the calculated SAR values.
    """
    # Initialize the SAR values with the closing prices as a starting point.
    sar_values = data['Close'].values
    
    # Iterate from the second row onwards to calculate the SAR.
    for i in range(1, len(sar_values)):
        # Apply the standard SAR formula.
        sar_values[i] = sar_values[i-1] + (acceleration_factor * (data.loc[i, 'Extreme Point'] - sar_values[i-1]))

    return sar_values

def _identify_parabolic_sar_indicators(data):
    """
    (Internal Helper) Generates trading signals based on the SAR's position relative to the price.

    Args:
        data (pd.DataFrame): DataFrame with 'SAR' and 'Close' columns.

    Returns:
        tuple[np.array, np.array, np.array]: A tuple containing signals for SAR being
                                             higher, lower, and for position changes.
    """
    # Signal for when the SAR is above or equal to the closing price (potential downtrend).
    higher_close = data.apply(lambda row: row['SAR'] >= row['Close'], axis=1).astype(int).values
    # Signal for when the SAR is below the closing price (potential uptrend).
    lower_close = data.apply(lambda row: row['SAR'] < row['Close'], axis=1).astype(int).values
    # Identify the exact point where the trend flips (SAR crosses the price).
    change_position = (higher_close[1:] != higher_close[:-1]).astype(int)
    change_position = np.concatenate(([0], change_position))
    
    return (higher_close, lower_close, change_position)

def identify_parabolic_sar_indicators(data, acceleration_factor=0.02):
    """
    Calculates a custom Parabolic SAR based on support and resistance trends.

    This function first identifies price trends using an external OBV-based module. It then
    calculates the Parabolic SAR and its related signals separately for periods of
    price resistance and price support, merging the results back into the original DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        acceleration_factor (float): The acceleration factor to be used in the SAR calculation.

    Returns:
        pd.DataFrame: The original DataFrame with Parabolic SAR signals added.
    """
    # Create a copy to avoid modifying the original DataFrame.
    temp_data = data.copy()

    # Step 1: Use an external module to identify initial support and resistance trends.
    temp_data = obv.identify_obv_indicators(temp_data)
    temp_data.dropna(subset=['Price Resistance Trend', 'Price Support Trend'], inplace=True)
    temp_data.reset_index(inplace=True, drop=True)

    # Step 2: Calculate SAR and signals for both resistance and support trends.
    trend_columns = ['Price Resistance Trend', 'Price Support Trend']
    for col in trend_columns:
        # Set the current trend being analyzed.
        temp_data['Current Trend'] = temp_data[col]
        # Group the consecutive trend periods.
        temp_data['Group Trend'] = _group_support_resistance_trend(temp_data)
        # Find the extreme point for each trend group.
        temp_data['Extreme Point'] = _calculate_extreme_point(temp_data)
        # Calculate the SAR value.
        temp_data['SAR'] = _calculate_sar(temp_data, acceleration_factor)
        # Generate signals based on the calculated SAR.
        higher_close, lower_close, change_position = _identify_parabolic_sar_indicators(temp_data)
        # Store the signals in uniquely named columns.
        temp_data[f'Higher {col} SAR'] = higher_close
        temp_data[f'Lower {col} SAR'] = lower_close
        temp_data[f'Change {col} SAR'] = change_position

    # Step 3: Define the final columns to be merged back into the original data.
    col_to_merge = [
        'Date',
        'Higher Price Resistance Trend SAR',
        'Lower Price Resistance Trend SAR',
        'Change Price Resistance Trend SAR',
        'Higher Price Support Trend SAR',
        'Lower Price Support Trend SAR',
        'Change Price Support Trend SAR',
    ] 
    # Merge the results back into the original DataFrame on the 'Date' key.
    data = pd.merge(data, temp_data[col_to_merge], on='Date', how='left')

    return data
