import numpy as np
import pandas as pd

def _calculate_daily_gains_or_losses(data):
    """
    (Internal Helper) Calculates and separates daily price changes into gains and losses.
    
    This function computes the percentage change from the previous day's close price
    and then splits the results into two separate lists: one for gains (positive changes)
    and one for losses (negative changes).

    Args:
        data (pd.DataFrame): DataFrame containing at least a 'Close' price column.

    Returns:
        tuple[list, list]: A tuple containing a list of gains and a list of losses.
    """
    # Calculate the daily percentage change in closing prices.
    gain_loss = 100 * (data['Close'].values[1:] - data['Close'].values[:-1]) / data['Close'].values[:-1]
    
    # Prepend a NaN value for the first day, as it has no prior day for comparison.
    gain_loss = np.concatenate(([np.nan], gain_loss))
    
    # Create a list for gains, keeping positive values and setting others to 0.
    gain = [val if np.isnan(val) else val if val > 0 else 0 for val in gain_loss]
    # Create a list for losses, keeping negative values and setting others to 0.
    loss = [val if np.isnan(val) else val if val < 0 else 0 for val in gain_loss]

    return (gain, loss)

def _calculate_historical_avg_gains_or_losses(data):
    """
    (Internal Helper) Calculates the 14-day simple moving average for gains and losses.

    Args:
        data (pd.DataFrame): DataFrame containing 'Gain' and 'Loss' columns.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple containing the average gains and average losses.
    """
    # Calculate the 14-day rolling average of gains.
    avg_gain = data['Gain'].rolling(14).mean()
    # Calculate the 14-day rolling average of losses and convert to a positive value.
    avg_loss = np.abs(data['Loss'].rolling(14).mean())

    return (avg_gain, avg_loss)
    
def _calculate_rsi(data):
    """
    (Internal Helper) Calculates the Relative Strength Index (RSI).
    
    Formula:
    $$ RSI = 100 - (100 / (1 + RS)) $$
    Where $ RS = (\text{Average Gain} / \text{Average Loss}) $.

    Args:
        data (pd.DataFrame): DataFrame with 'AVG Gain' and 'AVG Loss' columns.

    Returns:
        pd.Series: A pandas Series with the calculated RSI values.
    """
    # Calculate RSI using the standard formula.
    return 100 - (100 / (1 + (data['AVG Gain'] / data['AVG Loss'])))

def _identify_rsi_indicators(data):
    """
    (Internal Helper) Generates trading signals based on RSI values.

    This function identifies three conditions:
    1. Overbought: RSI is 70 or higher.
    2. Oversold: RSI is 30 or lower.
    3. Tendencies: A normalized value indicating bullish (>50) or bearish (<50) momentum.

    Args:
        data (pd.DataFrame): DataFrame with an 'RSI' column.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple with overbought, oversold, and tendency signals.
    """
    # Identify overbought conditions (RSI >= 70).
    rsi_overbought = data['RSI'].apply(lambda row: 1 if row >= 70 else 0)
    # Identify oversold conditions (RSI <= 30).
    rsi_oversold = data['RSI'].apply(lambda row: 1 if row <= 30 else 0)
    # Calculate tendency: a normalized value between -1 and 1 for the neutral range (30-70).
    rsi_tendencies = data['RSI'].apply(lambda row: (row - 50) / 20 if ((row > 30) & (row < 70)) else 0)

    return (rsi_overbought, rsi_oversold, rsi_tendencies)
    
def identify_rsi_indicators(data):
    """
    Calculates the Relative Strength Index (RSI) and its related trading indicators.

    This function orchestrates the entire RSI calculation process. It takes raw price data,
    calculates gains/losses, average gains/losses, the RSI itself, and finally generates
    overbought, oversold, and tendency signals, merging them back into the original DataFrame.

    Args:
        data (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.

    Returns:
        pd.DataFrame: The original DataFrame with added columns for RSI indicators.
    """
    # Create a copy of the data to avoid modifying the original DataFrame.
    temp_data = data.copy()

    # Step 1: Calculate daily gains and losses and add them to the DataFrame.
    gain_loss = _calculate_daily_gains_or_losses(temp_data)
    temp_data['Gain'] = gain_loss[0]
    temp_data['Loss'] = gain_loss[1]

    # Step 2: Calculate average gains and losses and add them to the DataFrame.
    avg_gain_loss = _calculate_historical_avg_gains_or_losses(temp_data)
    temp_data['AVG Gain'] = avg_gain_loss[0]
    temp_data['AVG Loss'] = avg_gain_loss[1]

    # Step 3: Calculate the RSI and add it to the DataFrame.
    temp_data['RSI'] = _calculate_rsi(temp_data)
    
    # Step 4: Identify RSI signals (overbought, oversold, tendencies) and add them.
    rsi_indicators = _identify_rsi_indicators(temp_data)
    temp_data['RSI Overbought'] = rsi_indicators[0]
    temp_data['RSI Oversold'] = rsi_indicators[1]
    temp_data['RSI Tendencies'] = rsi_indicators[2]

    # Clean up the data by removing initial rows where RSI could not be calculated.
    temp_data.dropna(subset=['RSI'], inplace=True)
    
    # Define the final indicator columns to be merged back into the original data.
    columns_to_keep = ['Date', 'RSI Overbought', 'RSI Oversold', 'RSI Tendencies']
    
    # Merge the results into the original DataFrame using the 'Date' as a key.
    data = pd.merge(data, temp_data[columns_to_keep], on='Date', how='left')
    
    return data