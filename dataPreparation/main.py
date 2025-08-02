import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from curl_cffi import requests
from sklearn.linear_model import LinearRegression

# This script assumes the existence of a 'technicalIndicators' module
# that contains a main function to generate all indicators.
from technicalIndicators.main import generate_all_technical_indicators

def download_stock_data(emiten, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.

    Args:
        emiten (str): The stock ticker symbol (e.g., 'BBCA').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """    
    # Download data from Yahoo Finance, appending '.JK' for the Jakarta Stock Exchange.
    session = requests.Session(impersonate="chrome")
    ticker = yf.Ticker(emiten, session=session)

    if start_date != '' or end_date != '':
        # Convert string dates to datetime objects.
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        data = ticker.history(start=start_date, end=end_date)
    else:
        data = ticker.history(period='max')
    
    # Drop the the columns Dividends, Stock Splits, and Capital Gains from the data
    for col in ['Dividends', 'Stock Splits', 'Capital Gains']:
        try:
            data.drop(columns=[col], inplace=True)
        except:
            pass
    
    # Move the 'Date' from the index to a column.
    data.reset_index(inplace=True)

    # Convert the Date format into YYYY-MM-DD
    data['Date'] = data['Date'].apply(lambda val: datetime.strptime(str(val).split(' ')[0], '%Y-%m-%d').date())

    return data

def _generate_linreg_gradient(target_data):
    """
    (Internal Helper) Calculates the slope of a data series without scaling.

    This function fits a linear regression model to a data series to determine
    its trend (slope). It forces the line to pass through the first data point
    by setting `fit_intercept=False`.

    Args:
        target_data (np.array): An array of numerical data.

    Returns:
        float: The calculated slope (gradient) of the regression line.
    """
    # Remove any NaN values from the input data.
    target_data = target_data[~np.isnan(target_data)]
    
    # Create an array representing the x-axis (time steps).
    X = np.linspace(0, len(target_data), len(target_data)).reshape(-1, 1)
    # Set the y-axis data relative to the first point to ensure the intercept is zero.
    y = target_data - target_data[0]
    
    # Create a linear regression model that does not fit an intercept.
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Return the calculated coefficient, which represents the slope.
    return model.coef_[0]

def _bin_linreg_gradients(val):
    """
    (Internal Helper) Classifies a slope value into a simple trend direction.

    Args:
        val (float): The slope value.

    Returns:
        str: 'Up Trend' for non-negative slopes, 'Down Trend' for negative slopes.
    """
    # Classify the trend as 'Down Trend' if the slope is negative.
    if val < 0:
        return 'Down Trend'
    # Otherwise, classify it as 'Up Trend'.
    else:
        return 'Up Trend'

def generate_all_linreg_gradients(data, target_column, rolling_window):
    """
    Generates a future trend label for each day based on a rolling window.

    For each day in the dataset, this function looks at the next `rolling_window`
    days, calculates the trend (slope) of that future period, and assigns a
    categorical label ('Up Trend' or 'Down Trend') to the current day.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the column to analyze (e.g., 'Close').
        rolling_window (int): The number of future days to look at for the trend.

    Returns:
        pd.DataFrame: The DataFrame with the new trend column added.
    """
    # Extract the target data series.
    target_data = data[target_column].values

    # Calculate the gradient for each rolling window of future data.
    linreg_gradients = [_generate_linreg_gradient(target_data[i+1:i+1+rolling_window]) for i in range(len(target_data)-rolling_window)]
    # Pad the end of the list with NaNs, as the trend cannot be calculated for the last few days.
    linreg_gradients = linreg_gradients + (np.ones(rolling_window) * np.nan).tolist()

    # Define the new column name.
    column = f'Upcoming {rolling_window} Days Trend'

    # Add the numerical gradients and the binned trend labels to the DataFrame.
    data[column] = linreg_gradients
    data[column] = data[column].apply(lambda val: _bin_linreg_gradients(val))

    return data

def prepare_data_for_modelling(emiten, start_date, end_date, target_column, rolling_windows, download=True):
    """
    Orchestrates the full data preparation pipeline for a machine learning model.

    This function loads data, generates future trend labels (the target variable),
    calculates a suite of technical indicators (the features), and cleans the
    final dataset by removing any rows with missing values.

    Args:
        emiten (str): The stock ticker symbol.
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
        target_column (str): The price column to use for trend calculation.
        rolling_windows (list): A list of integers for the different future trend windows to generate.

    Returns:
        pd.DataFrame: A clean DataFrame ready for model training.
    """
    # Load data by downloading from yahoo finance  API.
    if download:
        data = download_stock_data(emiten, start_date, end_date)
    else:
        data = pd.read_csv('dataPreparation/BBCA.csv')

    # Generate the future trend labels for each specified rolling window.
    for rolling_window in rolling_windows:
        data = generate_all_linreg_gradients(data, target_column, rolling_window)

    # Generate all technical indicators to be used as features.
    data = generate_all_technical_indicators(data)
    # Remove any rows that have NaN values after all calculations are complete.
    data.dropna(inplace=True)
    
    return data
