import numpy as np
import scipy.signal as sp

# These are assumed to be helper functions from your project structure.
from oldTechnicalIndicators.helpersFunctions.main import retrieve_linreg_gradients, group_trends

def _identify_general_peaks(price, peak_type, peak_distance, peak_rank_width):
    """
    (Internal Helper) Finds all potential peaks/troughs and ranks them by density.
    
    This function identifies all local maxima (resistance) or minima (support)
    and then assigns a rank to each one based on how many previous peaks
    are within a similar price range. This helps find clustered S/R zones.

    Args:
        price (np.array): Array of price data.
        peak_type (str): 'resistance' to find peaks or 'support' to find troughs.
        peak_distance (int): Minimum horizontal distance between peaks.
        peak_rank_width (float): The price width to consider peaks as being at a similar level.

    Returns:
        dict: A dictionary mapping peak indices to their calculated rank.
    """
    # Use SciPy's find_peaks to locate local maxima (resistance).
    if peak_type == 'resistance':
        peaks, _ = sp.find_peaks(price, distance=peak_distance)
    # To find local minima (support), we invert the price series and find its peaks.
    elif peak_type == 'support':
        peaks, _ = sp.find_peaks(-1 * price, distance=peak_distance)
        
    # Initialize a dictionary to store the rank of each identified peak.
    peak_to_rank = {peak: 0 for peak in peaks}

    # Iterate through each peak to compare it with previous peaks.
    for i, current_peak in enumerate(peaks):
        current_price = price[current_peak]
        # Check all peaks that came before the current one.
        for previous_peak in peaks[:i]:
            # If a previous peak's price is close to the current peak's price, increment the rank.
            if abs(current_price - price[previous_peak]) <= peak_rank_width:
                peak_to_rank[current_peak] += 1

    return peak_to_rank

def _identify_strong_peak(price, strong_peak_distance, strong_peak_prominence):
    """
    (Internal Helper) Finds only the most significant peaks using prominence.

    Peak prominence measures how much a peak stands out from the surrounding
    data, making it an excellent filter for major market turning points.

    Args:
        price (np.array): Array of price data.
        strong_peak_distance (int): Minimum horizontal distance between strong peaks.
        strong_peak_prominence (float): The required prominence for a peak to be considered 'strong'.

    Returns:
        list: A list of the price values at the identified strong peaks.
    """
    # Use find_peaks with the 'prominence' argument to filter for major peaks.
    strong_peaks, _ = sp.find_peaks(
      price,
      distance=strong_peak_distance,
      prominence=strong_peak_prominence
    )
    
    # Get the actual price values at the locations of the strong peaks.
    strong_peaks_values = price[strong_peaks].tolist()

    return strong_peaks_values

def _combine_general_and_strong_peaks(price, peak_type, peak_to_rank, strong_peaks_values, resistance_min_pivot_rank, peak_rank_width):
    """
    (Internal Helper) Merges strong peaks with high-rank general peaks.

    This function creates a final list of significant price levels by combining
    all strong peaks with any general peaks that have a high enough rank. It then
    groups very close price levels together into a single representative value.

    Args:
        price (np.array): Array of price data.
        peak_type (str): 'resistance' or 'support'.
        peak_to_rank (dict): Dictionary of ranked general peaks.
        strong_peaks_values (list): List of prices for strong peaks.
        resistance_min_pivot_rank (int): The minimum rank for a general peak to be included.
        peak_rank_width (float): The price width used for grouping nearby peaks.

    Returns:
        list: A final, consolidated list of significant peak price values.
    """
    # Start the final list with all identified strong peaks.
    final_peaks = strong_peaks_values
    
    # Add any general peaks that meet the minimum rank requirement.
    for peak, rank in peak_to_rank.items():
        if rank >= resistance_min_pivot_rank:
            final_peaks.append(price[peak])
    
    # Sort the combined list of peak prices.
    final_peaks.sort()
    
    # Group peaks that are very close in price into "bins".
    final_peak_bins = []
    current_bin = [final_peaks[0]]
    
    for r in final_peaks:
        # If the next peak is close to the last one in the current bin, add it.
        if r - current_bin[-1] < peak_rank_width:
            current_bin.append(r)
        # Otherwise, finalize the current bin and start a new one.
        else:
            final_peak_bins.append(current_bin)
            current_bin = [r]
    
    # Append the last bin to the list.
    final_peak_bins.append(current_bin)

    # For each bin, select a single representative price.
    if peak_type == 'resistance':
        # For resistance zones, use the highest price in the cluster.
        final_peaks = [np.max(bin) for bin in final_peak_bins]
    elif peak_type == 'support':
        # For support zones, use the lowest price in the cluster.
        final_peaks = [np.min(bin) for bin in final_peak_bins]

    return final_peaks

def _identify_peaks_index(prices, final_peaks):
    """
    (Internal Helper) Finds the array indices corresponding to the final peak prices.

    Args:
        prices (np.array): The original array of price data.
        final_peaks (list): The consolidated list of peak price values.

    Returns:
        list: A list of indices corresponding to the locations of the final peaks.
    """
    start_index = 0
    final_peaks_index = []
    
    # For each final peak price, find its first occurrence in the price array.
    for peak in final_peaks:
        # Search from the last found index to ensure we find unique peak occurrences in order.
        for i, val in enumerate(prices[start_index:], start=start_index):
            if val == peak:
                final_peaks_index.append(i)
                break
    
    # Assertions to ensure data integrity: every peak price found a unique index.
    assert len(final_peaks_index) == len(final_peaks)
    assert len(np.unique(final_peaks_index)) == len(final_peaks_index)

    return final_peaks_index

def identify_all_peaks(price, peak_type, peak_distance=2, peak_rank_width=2, resistance_min_pivot_rank=3, strong_peak_distance=60, strong_peak_prominence=20):
    """
    Orchestrates the entire process of identifying significant support or resistance levels.

    Args:
        price (np.array): Array of price data.
        peak_type (str): 'resistance' or 'support'.
        (Other args are passed to helper functions).

    Returns:
        list: A sorted list of the indices of all significant peaks.
    """
    # Step 1: Find and rank all general peaks.
    peak_to_rank = _identify_general_peaks(price, peak_type, peak_distance, peak_rank_width)

    # Step 2: Find all strong, prominent peaks.
    strong_peaks_values = _identify_strong_peak(price, strong_peak_distance, strong_peak_prominence)

    # Step 3: Combine strong and high-rank general peaks into a final list of price levels.
    final_peaks = _combine_general_and_strong_peaks(price, peak_type, peak_to_rank, strong_peaks_values, resistance_min_pivot_rank, peak_rank_width)

    # Step 4: Find the array indices for these final price levels.
    final_peaks_index = _identify_peaks_index(price, final_peaks)
    final_peaks_index.sort()
        
    return final_peaks_index

def identify_current_trends_on_peaks(price, peaks_index, n_peaks):
    """
    Calculates the trend slope using the last N peaks relative to the current price.

    For each day, this function looks at the last `n_peaks` S/R levels, adds the
    current day's price, and calculates a linear regression slope to quantify the
    most recent trend.

    Args:
        price (np.array): Array of price or volume data.
        peaks_index (list): A sorted list of indices for the S/R levels.
        n_peaks (int): The number of recent peaks to include in the trend calculation.

    Returns:
        np.array: An array containing the trend classification for each day.
    """
    start_index = 0
    inside_trends = []

    # Iterate through each price point (each day) in the dataset.
    for i, p in enumerate(price):
        # Advance the 'start_index' to keep track of which peak is next.
        if peaks_index[np.min([start_index, len(peaks_index)-1])] < i:
            start_index += 1
        
        # Get the last N peaks that occurred before the current day 'i'.
        current_peaks_index = peaks_index[start_index-n_peaks:start_index]
        
        # If we don't have enough historical peaks yet, the trend is undefined.
        if len(current_peaks_index) < n_peaks:
                inside_trends.append(np.nan)    
        else:
            # Create a data series of the prices at the last N peaks plus the current day's price.
            current_peaks = np.concatenate((price[current_peaks_index], [p]))
            # Calculate the linear regression gradient (slope) of this series.
            inside_trends.append(retrieve_linreg_gradients(current_peaks))

    # Convert the numerical slope values into trend categories ('up', 'down', 'sideways').
    inside_trends = [group_trends(val, 0.1) for val in inside_trends]        
    
    return np.array(inside_trends)

def identify_obv_indicators(data):
    """
    Identifies price-volume trends and divergences using support and resistance.

    This main function calculates trends for both price and volume between key S/R
    levels. A divergence occurs when price and volume are not in agreement (e.g.,
    price trend is up, but volume trend is down), which can signal a potential reversal.

    Args:
        data (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns.

    Returns:
        pd.DataFrame: The original DataFrame with four new trend-analysis columns.
    """
    # Identify all significant resistance and support points in the closing price.
    resistances_index = identify_all_peaks(data['Close'].values, 'resistance')
    supports_index = identify_all_peaks(data['Close'].values, 'support')

    # Added the last index of the data as the resistance and supports points 
    resistances_index.append(len(data)-1)
    supports_index.append(len(data)-1)
    
    # Calculate the trend of the PRICE between the last 3 resistance points.
    price_resistance_trend = identify_current_trends_on_peaks(data['Close'].values, resistances_index, 3)
    # Calculate the trend of the PRICE between the last 3 support points.
    price_support_trend = identify_current_trends_on_peaks(data['Close'].values, supports_index, 3)
    # Calculate the trend of the VOLUME between the last 3 resistance points.
    volume_resistance_trend = identify_current_trends_on_peaks(data['Volume'].values, resistances_index, 3)
    # Calculate the trend of the VOLUME between the last 3 support points.
    volume_support_trend = identify_current_trends_on_peaks(data['Volume'].values, supports_index, 3)

    # Add the newly calculated trend data as columns to the DataFrame.
    data['Price Resistance Trend'] = price_resistance_trend
    data['Price Support Trend'] = price_support_trend
    data['Volume Resistance Trend'] = volume_resistance_trend
    data['Volume Suppport Trend'] = volume_support_trend

    return data