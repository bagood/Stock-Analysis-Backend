import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def identify_historical_trends(data, column, rolling_window, make_bool_up=None, make_bool_down=None):
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
    linreg_gradients = np.array([np.nan if i < rolling_window else _retrieve_linreg_gradients(data[column].values[i-rolling_window:i]) for i in range(len(data))])
    
    if make_bool_up:
        return np.array([np.nan if np.isnan(val) else val > 0 for val in linreg_gradients]).astype(int)
    elif make_bool_down:
        return np.array([np.nan if np.isnan(val) else val < 0 for val in linreg_gradients]).astype(int)
    return linreg_gradients

def _retrieve_linreg_gradients(target_data):
    """
    Calculates the slope (gradient) of a data series using linear regression.

    This function takes a series of data points, scales them to a 0-1 range,
    fits a linear regression model, and returns the slope of the regression line.
    This is used to quantify the trend of the data.

    Args:
        target_data (np.array): An array of numerical data.

    Returns:
        float: The calculated slope of the linear regression line. Returns 0
               if there are not enough data points to calculate a trend.
    """
    target_data = target_data[~np.isnan(target_data)]
    
    if len(target_data) > 1:
        scaler = MinMaxScaler()
        
        X = np.linspace(0, len(target_data), len(target_data)).reshape(-1, 1)
        y = scaler.fit_transform(target_data.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X, y)
    
        return model.coef_[0, 0]
    else:
        return 0

def backfill_inf_values(data, column):
    """
    Replaces infinite values in a DataFrame column with the preceding max or min value.

    This function finds any `np.inf` or `-np.inf` values and replaces them with
    the maximum or minimum value found in the column up to that point, respectively.

    Args:
        data (pd.DataFrame): The DataFrame to process.
        column (str): The name of the column to clean.

    Returns:
        pd.DataFrame: The DataFrame with infinite values replaced.
    """
    inf_index = data.loc[data[column].isin([np.inf, -np.inf]), :].index.tolist()
    
    for index in inf_index:
        if data.loc[index, column] < 0:
            data.loc[index, column] = np.min(data[column][:index])
        else:
            data.loc[index, column] = np.max(data[column][:index])

    assert not data[column].isin([np.inf, -np.inf]).any()

    return data

def scale_indicators(data, column):
    """
    Scales the values of a DataFrame column to a range between 0 and 1.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to scale.

    Returns:
        np.array: An array of the scaled values.
    """
    scaler = MinMaxScaler()
    
    return scaler.fit_transform(data[column].values.reshape(-1, 1)).T[0]

def group_trends(trends, threshold):
    """
    Classifies a trend value as 'up' (1), 'down' (-1), or 'sideways' (0).

    Args:
        trends (float): The numerical trend value (e.g., a slope).
        threshold (float): The value below which a trend is considered 'sideways'.

    Returns:
        int: 1 for an uptrend, -1 for a downtrend, and 0 for a sideways trend.
    """
    if np.abs(trends) <= threshold:
        return 0
    else:
        return np.abs(trends) / trends