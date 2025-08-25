import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from curl_cffi import requests
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def _download_stock_data(emiten: str, start_date: str, end_date: str) -> pd.DataFrame: 
    """
    (Internal Helper) Downloads historical stock data from Yahoo Finance for a given ticker.

    This function fetches daily 'Open', 'High', 'Low', 'Close', and 'Volume' data.
    It automatically appends the '.JK' suffix, which is standard for tickers
    on the Jakarta Stock Exchange (IDX). It also performs basic data cleaning
    by removing non-essential columns and standardizing the date format.

    Args:
        emiten (str): The stock ticker symbol (e.g., 'BBCA').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
                          If empty, the download will start from the earliest available date.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
                        If empty, the download will go up to the most recent date.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned historical stock data,
                      or None if the download fails.
    """
    session = requests.Session(impersonate="chrome123")
    ticker = yf.Ticker(f"{emiten}.JK", session=session)

    start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.strptime('2021-01-01', '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
    data = ticker.history(start=start, end=end)

    columns_to_drop = ['Dividends', 'Stock Splits', 'Capital Gains']
    for col in columns_to_drop:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date

    return data
    
def _generate_linreg_gradient(target_data: np.array) -> float:
    """
    (Internal Helper) Calculates the slope of a data series using linear regression.

    This function fits a line to the provided data points and returns the slope
    (gradient). This slope represents the trend of the data over the given period.
    The intercept is forced to zero by centering the data, focusing solely on the trend.

    Args:
        target_data (np.array): A numpy array of numerical data (e.g., closing prices).

    Returns:
        float: The calculated slope (gradient) of the regression line.
    """
    X = np.arange(len(target_data)).reshape(-1, 1)
    y = target_data - target_data[0]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    linreg_gradient = model.coef_[0]
    
    return linreg_gradient

def _generate_linreg_rsquared_score(target_data: np.array) -> float:
    """
    (Internal Helper)

    Args:
        target_data (np.array): A numpy array of numerical data (e.g., closing prices).

    Returns:
        float:
    """
    X = np.arange(len(target_data)).reshape(-1, 1)
    y = target_data - target_data[0]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    rsquared_score = r2_score(y, y_pred)
    
    return rsquared_score

def _bin_linreg_gradients(val: float) -> str:
    """
    (Internal Helper) Classifies a slope value into a categorical trend direction.

    Args:
        val (float): The slope value from the linear regression calculation.

    Returns:
        str: 'Up Trend' for positive or zero slopes, 'Down Trend' for negative slopes,
             or NaN if the input is NaN.
    """
    if np.isnan(val):
        return val
    if val < 0:
        return 'Down Trend'
    else:
        return 'Up Trend'
    
def _bin_linreg_rsquared_score(val: float) -> str:
    """
    (Internal Helper)

    Args:
        val (float):

    Returns:
        str:
    """
    if np.isnan(val):
        return val
    if val <= 0.45:
        return 'Weak Trend'
    else:
        return 'Strong Trend'

def _generate_all_linreg_gradients(data: pd.DataFrame, target_column: str, rolling_window: int) -> pd.DataFrame:
    """
    (Internal Helper) Generates a future trend label for each day based on a rolling window.

    For each day in the dataset, this function looks at the *next* `rolling_window`
    days of the `target_column`, calculates the trend (slope) for that future period,
    and assigns a categorical label ('Up Trend' or 'Down Trend') to the current day.
    This resulting column is the target variable for the machine learning model.

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data.
        target_column (str): The name of the column to analyze (e.g., 'Close').
        rolling_window (int): The number of future days to look at for the trend.

    Returns:
        pd.DataFrame: The DataFrame with the new future trend column added.
    """
    column_name = f'Upcoming {rolling_window} Days Trend'    
    target_data = data[target_column].values

    linreg_gradients = [
        _generate_linreg_gradient(target_data[i : i+rolling_window])
        for i in range(len(target_data) - rolling_window)
    ]

    padding = [np.nan] * rolling_window
    full_gradient_list = linreg_gradients + padding

    data[column_name] = full_gradient_list
    data[column_name] = data[column_name].apply(_bin_linreg_gradients)
    
    return data

def _generate_all_linreg_rsquared_score(data: pd.DataFrame, target_column: str, rolling_window: int) -> pd.DataFrame:
    """
    (Internal Helper)

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data.
        target_column (str): The name of the column to analyze (e.g., 'Close').
        rolling_window (int): The number of future days to look at for the trend.

    Returns:
        pd.DataFrame:
    """
    column_name = f'Upcoming {rolling_window} Days Strength'    
    target_data = data[target_column].values

    linreg_rsquared_score = [
        _generate_linreg_rsquared_score(target_data[i : i+rolling_window])
        for i in range(len(target_data) - rolling_window)
    ]

    padding = [np.nan] * rolling_window
    full_rsquared_score_list = linreg_rsquared_score + padding

    data[column_name] = full_rsquared_score_list
    data[column_name] = data[column_name].apply(_bin_linreg_rsquared_score)
    
    return data