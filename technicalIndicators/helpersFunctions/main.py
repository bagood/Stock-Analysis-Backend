import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def retrieve_linreg_gradients(target_data):
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
    # Remove any NaN values from the input data to prevent errors.
    target_data = target_data[~np.isnan(target_data)]
    
    # Ensure there is more than one data point to form a line.
    if len(target_data) > 1:
        # Initialize the Min-Max Scaler to normalize data.
        scaler = MinMaxScaler()
        
        # Create an array representing the x-axis (time steps).
        X = np.linspace(0, len(target_data), len(target_data)).reshape(-1, 1)
        # Scale the y-axis data (target_data) to a range between 0 and 1.
        y = scaler.fit_transform(target_data.reshape(-1, 1))
        
        # Create and fit the linear regression model.
        model = LinearRegression()
        model.fit(X, y)
    
        # Return the coefficient (slope) of the fitted line.
        return model.coef_[0, 0]
    else:
        # If there's not enough data, return 0, indicating no trend.
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
    # Get the index of all rows containing an infinite value in the specified column.
    inf_index = data.loc[data[column].isin([np.inf, -np.inf]), :].index.tolist()
    
    # Iterate through the identified indices.
    for index in inf_index:
        # If the value is negative infinity, replace it with the minimum value seen so far.
        if data.loc[index, column] < 0:
            data.loc[index, column] = np.min(data[column][:index])
        # If the value is positive infinity, replace it with the maximum value seen so far.
        else:
            data.loc[index, column] = np.max(data[column][:index])

    # Assert that no infinite values remain in the column.
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
    # Initialize the Min-Max Scaler.
    scaler = MinMaxScaler()
    
    # Reshape the column data, apply the scaler, and return the transformed array.
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
    # If the absolute trend value is within the threshold, classify it as sideways (0).
    if np.abs(trends) <= threshold:
        return 0
    # Otherwise, return 1 for positive trends and -1 for negative trends.
    else:
        return np.abs(trends) / trends