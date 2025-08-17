import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from curl_cffi import requests
from sklearn.linear_model import LinearRegression

from technicalIndicators.main import generate_all_technical_indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def download_stock_data(emiten: str, start_date: str, end_date: str):
    """
    Downloads historical stock data from Yahoo Finance for a given ticker.

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
    logging.info(f"Starting stock data download for ticker: {emiten}.JK")

    try:
        session = requests.Session(impersonate="chrome123")
        ticker = yf.Ticker(f"{emiten}.JK", session=session)

        if not start_date and not end_date:
            logging.info("No start or end date provided. Fetching maximum available historical data.")
            data = ticker.history(period='max')
        else:
            start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now()
            end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            
            logging.info(f"Fetching data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}.")
            data = ticker.history(start=start, end=end)

        if data.empty:
            logging.warning(f"No data returned for ticker {emiten}.JK. It may be an invalid ticker or have no data for the period.")
            return None

        logging.info(f"Successfully downloaded {len(data)} rows of data.")

        columns_to_drop = ['Dividends', 'Stock Splits', 'Capital Gains']
        for col in columns_to_drop:
            if col in data.columns:
                data.drop(columns=[col], inplace=True)
                logging.debug(f"Dropped column: '{col}'")

        data.reset_index(inplace=True)

        data['Date'] = data['Date'].dt.date
        logging.info("Data cleaning and formatting complete.")

        return data

    except Exception as e:
        logging.error(f"An error occurred while downloading data for {emiten}.JK: {e}")
        return None

def _generate_linreg_gradient(target_data: np.array):
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
    target_data = target_data[~np.isnan(target_data)]

    if len(target_data) < 2:
        return np.nan
    
    X = np.arange(len(target_data)).reshape(-1, 1)
    y = target_data - target_data[0]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    linreg_gradient = model.coef_[0]
    
    return linreg_gradient

def _bin_linreg_gradients(val: float):
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

def generate_all_linreg_gradients(data: pd.DataFrame, target_column: str, rolling_window: int):
    """
    Generates a future trend label for each day based on a rolling window.

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
    logging.info(f"Generating future trend labels for a {rolling_window}-day window using '{target_column}' column.")
    
    column_name = f'Upcoming {rolling_window} Days Trend'    
    target_data = data[target_column].values

    linreg_gradients = [
        _generate_linreg_gradient(target_data[i+1 : i+1+rolling_window])
        for i in range(len(target_data) - rolling_window)
    ]

    padding = [np.nan] * rolling_window
    full_gradient_list = linreg_gradients + padding

    data[column_name] = full_gradient_list
    data[column_name] = data[column_name].apply(_bin_linreg_gradients)
    
    logging.info(f"Finished generating trend labels. Added column: '{column_name}'.")

    return data

def prepare_data_for_modelling(emiten: str, start_date: str, end_date: str, target_column: str, rolling_windows: list, download: bool = True):
    """
    Orchestrates the full data preparation pipeline for a machine learning model.

    This function serves as the main controller, executing a sequence of steps:
    1. Downloads or loads historical stock data.
    2. Generates a comprehensive set of technical indicators to be used as model features.
    3. Creates the target variable(s) by calculating future trend labels.
    4. Cleans the final dataset by removing any rows with missing values (NaNs)
       that result from the indicator and trend calculations.

    Args:
        emiten (str): The stock ticker symbol.
        start_date (str): The start date for the data ('YYYY-MM-DD').
        end_date (str): The end date for the data ('YYYY-MM-DD').
        target_column (str): The price column (e.g., 'Close') to use for trend calculation.
        rolling_windows (list): A list of integers for the future trend windows (e.g., [5, 10]).
        download (bool): If True, downloads data from Yahoo Finance. If False, loads a local dummy file.

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation.
    """
    logging.info(f"--- Starting Data Preparation Pipeline for Ticker: {emiten} ---")

    if download:
        data = download_stock_data(emiten, start_date, end_date)
    else:
        logging.info("Loading data from local 'dummy_data.csv' file.")
        data = pd.read_csv('dataPreparation/dummy_data.csv')
    
    if data is None or data.empty:
        logging.error("Data loading failed. Aborting pipeline.")
        return pd.DataFrame()

    logging.info("Generating technical indicators as features...")
    data = generate_all_technical_indicators(data)
    
    logging.info("Generating future trend labels as target variables...")
    for window in rolling_windows:
        data = generate_all_linreg_gradients(data, target_column, window)
    
    data.dropna(subset=[f'Upcoming {window} Days Trend' for window in rolling_windows], inplace=True)

    logging.info(f"--- Data Preparation Pipeline for {emiten} Finished Successfully ---")
    return data


def _prepare_data_for_forecasting(emiten: str, start_date: str, end_date: str, rolling_window: int, download: bool = True):
    """
    Orchestrates the data preparation pipeline for making forecasts using the developed machine learning model.

    This function serves as the helper, executing a sequence of steps:
    1. Downloads or loads historical stock data.
    2. Generates a comprehensive set of technical indicators to be used as model features.
    3. Gets the tail of the data for the forecasting data

    Args:
        emiten (str): The stock ticker symbol.
        start_date (str): The start date for the data ('YYYY-MM-DD').
        end_date (str): The end date for the data ('YYYY-MM-DD').
        rolling_window (int): An integers for the future trend windows, correlates with the total of forecasting data
        download (bool): If True, downloads data from Yahoo Finance. If False, loads a local dummy file.

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation.
    """
    if download:
        data = download_stock_data(emiten, start_date, end_date)
    else:
        logging.info("Loading data from local 'dummy_data.csv' file.")
        data = pd.read_csv('dataPreparation/dummy_data.csv')
    
    if data is None or data.empty:
        logging.error("Data loading failed. Aborting pipeline.")
        return pd.DataFrame()

    logging.info("Generating technical indicators as features...")
    data = generate_all_technical_indicators(data)
    
    logging.info("Dropping all rows without any NaN values to create a clean forecasting dataset.")
    forecasting_data = data.tail(rolling_window)

    return forecasting_data

def prepare_data_for_forecasting(list_of_emitens: list, start_date: str, end_date: str, rolling_window: int, download: bool = True):
    """
    Orchestrates the full data preparation pipeline for making forecasts using the developed machine learning model.

    Args:
        emiten (str): The stock ticker symbol.
        start_date (str): The start date for the data ('YYYY-MM-DD').
        end_date (str): The end date for the data ('YYYY-MM-DD').
        rolling_window (int): An integers for the future trend windows, correlates with the total of forecasting data
        download (bool): If True, downloads data from Yahoo Finance. If False, loads a local dummy file.

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation.
    """
    logging.info(f"--- Starting Data Preparation Pipeline for {len(list_of_emitens)} Tickers ---")
    
    all_emiten_data = pd.DataFrame()
    total_failed_stocks = 0

    for emiten in list_of_emitens:
        try:
            logging.info(f"--- Starting Data Preparation Pipeline for Ticker: {emiten} ---")

            emiten_data = _prepare_data_for_forecasting(emiten, start_date, end_date, rolling_window, download)
            emiten_data['Kode'] = emiten
            all_emiten_data = pd.concat((all_emiten_data, emiten_data))

            logging.info(f"--- Data Preparation Pipeline for {emiten} Finished Successfully ---")
        except:
            total_failed_stocks += 1
            logging.warning(f"--- Data Preparation Pipeline for {emiten} Failed ---")

    logging.info(f"--- Succesfully Prepare the Forecasting Data for {len(list_of_emitens) - total_failed_stocks} out of {len(list_of_emitens)} Stocks ---")

    return all_emiten_data