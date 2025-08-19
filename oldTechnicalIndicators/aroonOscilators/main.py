import numpy as np
import pandas as pd

def _calculate_aroon_direction(data, direction):
    """
    (Internal Helper) Calculates the Aroon Up or Aroon Down indicator value.

    The Aroon indicator measures the number of periods since the price last
    recorded a 25-period high or low. This implementation calculates that value
    and scales the result to a range between 0 and 1.

    Args:
        data (pd.DataFrame): The input DataFrame with 'High' and 'Low' columns.
        direction (str): 'up' to calculate Aroon Up, or 'down' for Aroon Down.

    Returns:
        np.array: An array of the calculated Aroon Up or Aroon Down values.
    """
    # Calculate Aroon Up based on the number of periods since the last 25-period high.
    if direction == 'up':
        # For each day 'i', look at the previous 25 days.
        # np.where finds the index of the highest high in that 25-day window.
        # (25 - index) calculates how many periods ago that high occurred.
        # The result is scaled by (4 / 100), which is equivalent to (1/25),
        # normalizing the output to a 0-1 range instead of the standard 0-100.
        return 4 * (25 - np.array([np.nan if i < 25 else np.where(data['High'].values[i-25:i] == np.max(data['High'].values[i-25:i]))[0][0] for i in range(len(data))])) / 100
    
    # Calculate Aroon Down based on the number of periods since the last 25-period low.
    elif direction == 'down':
        # The logic is the same as Aroon Up, but it searches for the lowest low.
        return 4 * (25 - np.array([np.nan if i < 25 else np.where(data['Low'].values[i-25:i] == np.min(data['Low'].values[i-25:i]))[0][0] for i in range(len(data))])) / 100

def identify_ao_indicators(data):
    """
    Calculates the Aroon Up, Aroon Down, and the final Aroon Oscillator.

    The Aroon Oscillator is a trend-following indicator that uses the Aroon Up
    and Aroon Down lines to gauge the strength of a current trend and the
    likelihood that it will continue.

    - Aroon Up: Measures the strength of the uptrend.
    - Aroon Down: Measures the strength of the downtrend.
    - Aroon Oscillator: The difference between Aroon Up and Aroon Down. A positive
      value indicates an uptrend, while a negative value indicates a downtrend.

    Args:
        data (pd.DataFrame): The input DataFrame with 'High' and 'Low' columns.

    Returns:
        pd.DataFrame: The original DataFrame with 'Aroon Up', 'Aroon Down',
                      and 'Aaron' (Oscillator) columns added.
    """
    # Step 1: Calculate the Aroon Up line and add it to the DataFrame.
    data['Aroon Up'] = _calculate_aroon_direction(data, 'up')
    
    # Step 2: Calculate the Aroon Down line and add it to the DataFrame.
    data['Aroon Down'] = _calculate_aroon_direction(data, 'down')
    
    # Step 3: Calculate the Aroon Oscillator by subtracting Aroon Down from Aroon Up.
    data['Aaron'] = data['Aroon Up'] - data['Aroon Down']

    return data
