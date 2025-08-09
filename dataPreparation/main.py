import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from curl_cffi import requests
from sklearn.linear_model import LinearRegression

# This script assumes the existence of a 'technicalIndicators' module
# that contains a main function to generate all indicators.
from technicalIndicators.main import generate_all_technical_indicators

# --- Logging Configuration ---
# Configure the logger to display the timestamp, log level, and message.
# This helps in tracking the script's execution flow and debugging.
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
        # Use a requests session with browser-like headers to improve reliability
        # and avoid potential blocking from the data source.
        session = requests.Session(impersonate="chrome123")
        ticker = yf.Ticker(f"{emiten}.JK", session=session)

        # Determine the period for data download based on provided dates.
        if not start_date and not end_date:
            logging.info("No start or end date provided. Fetching maximum available historical data.")
            data = ticker.history(period='max')
        else:
            # Convert string dates to datetime objects for the API call.
            # Default to the current time if a date is missing.
            start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now()
            end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            logging.info(f"Fetching data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}.")
            data = ticker.history(start=start, end=end)

        if data.empty:
            logging.warning(f"No data returned for ticker {emiten}.JK. It may be an invalid ticker or have no data for the period.")
            return None

        logging.info(f"Successfully downloaded {len(data)} rows of data.")

        # --- Data Cleaning ---
        # Drop columns that are not needed for technical analysis.
        columns_to_drop = ['Dividends', 'Stock Splits', 'Capital Gains']
        for col in columns_to_drop:
            if col in data.columns:
                data.drop(columns=[col], inplace=True)
                logging.debug(f"Dropped column: '{col}'")

        # Move the 'Date' from the DataFrame's index to a regular column.
        data.reset_index(inplace=True)

        # Standardize the 'Date' column to a simple 'YYYY-MM-DD' date format.
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
    # Ensure the input data has no NaN values, which would cause the model to fail.
    target_data = target_data[~np.isnan(target_data)]

    # If the array is empty after cleaning, no slope can be calculated.
    if len(target_data) < 2:
        return np.nan

    # Create an array representing the x-axis (time steps, e.g., 0, 1, 2, ...).
    # This must be a 2D array for scikit-learn's model.
    X = np.arange(len(target_data)).reshape(-1, 1)

    # Set the y-axis data. We subtract the first point to make the intercept effectively zero.
    # This ensures the model only calculates the trend (slope) from the starting point.
    y = target_data - target_data[0]

    # Initialize and fit a linear regression model.
    # `fit_intercept=False` tells the model the line must pass through the origin (0,0).
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # The model's coefficient is the calculated slope of the trend line.
    return model.coef_[0]

def _bin_linreg_gradients(val: float):
    """
    (Internal Helper) Classifies a slope value into a categorical trend direction.

    Args:
        val (float): The slope value from the linear regression calculation.

    Returns:
        str: 'Up Trend' for positive or zero slopes, 'Down Trend' for negative slopes,
             or NaN if the input is NaN.
    """
    # Propagate NaN values without attempting to classify them.
    if np.isnan(val):
        return val

    # A negative slope indicates a downward trend.
    if val < 0:
        return 'Down Trend'
    # A positive or zero slope indicates an upward or flat trend, classified as 'Up Trend'.
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
    
    # Extract the target data series into a numpy array for efficient processing.
    target_data = data[target_column].values

    # Calculate the gradient for each rolling window of *future* data using a list comprehension.
    # For each day `i`, it looks at the slice from `i+1` to `i+1+rolling_window`.
    linreg_gradients = [
        _generate_linreg_gradient(target_data[i+1 : i+1+rolling_window])
        for i in range(len(target_data) - rolling_window)
    ]

    # The trend cannot be calculated for the last `rolling_window` days of the dataset,
    # as there isn't enough future data. We pad the end of the list with NaNs.
    padding = [np.nan] * rolling_window
    full_gradient_list = linreg_gradients + padding

    # Add the binned trend labels to the DataFrame.
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

    # --- Step 1: Load Raw Data ---
    if download:
        data = download_stock_data(emiten, start_date, end_date)
    else:
        logging.info("Loading data from local 'dummy_data.csv' file.")
        data = pd.read_csv('dataPreparation/dummy_data.csv')
    
    if data is None or data.empty:
        logging.error("Data loading failed. Aborting pipeline.")
        return pd.DataFrame() # Return empty DataFrame on failure

    # --- Step 2: Generate Features (Technical Indicators) ---
    logging.info("Generating technical indicators as features...")
    data = generate_all_technical_indicators(data)
    logging.info(f"Data shape after adding indicators: {data.shape}")

    # --- Step 3: Generate Target Variable (Future Trend Labels) ---
    logging.info("Generating future trend labels as target variables...")
    for window in rolling_windows:
        data = generate_all_linreg_gradients(data, target_column, window)

    # --- Step 4: Final Cleaning ---
    # Many indicators (e.g., moving averages) and the target labels create NaN
    # values at the beginning and end of the dataset. These rows must be removed.
    initial_rows = len(data)
    logging.info("Dropping all rows with any NaN values to create a clean dataset.")
    data.dropna(inplace=True)
    final_rows = len(data)
    logging.info(f"Removed {initial_rows - final_rows} rows. Final dataset shape: {data.shape}.")

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
    # --- Step 1: Load Raw Data ---
    if download:
        data = download_stock_data(emiten, start_date, end_date)
    else:
        logging.info("Loading data from local 'dummy_data.csv' file.")
        data = pd.read_csv('dataPreparation/dummy_data.csv')
    
    if data is None or data.empty:
        logging.error("Data loading failed. Aborting pipeline.")
        return pd.DataFrame() # Return empty DataFrame on failure

    # --- Step 2: Generate Features (Technical Indicators) ---
    logging.info("Generating technical indicators as features...")
    data = generate_all_technical_indicators(data)
    logging.info(f"Data shape after adding indicators: {data.shape}")

    # --- Step 3: Final Cleaning ---
    # Select the rolling_window amount from the tail of the data
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
    # Creates an empty dataframe to store all emitens' forecast data
    all_emiten_data = pd.DataFrame()

    # Calculate the total of stocks failed in being processed
    total_failed_stocks = 0

    # Iterate over each emiten to acquire the forecast data
    for emiten in list_of_emitens:
        try:
            logging.info(f"--- Starting Data Preparation Pipeline for Ticker: {emiten} ---")
            emiten_data = _prepare_data_for_forecasting(emiten, start_date, end_date, rolling_window, download)
            
            # Added the kode for the emiten on the data
            emiten_data['Kode'] = emiten

            # Combine the current forecasting data with the other forecasting data
            all_emiten_data = pd.concat((all_emiten_data, emiten_data))
            logging.info(f"--- Data Preparation Pipeline for {emiten} Finished Successfully ---")
        except:
            total_failed_stocks += 1
            logging.info(f"--- Data Preparation Pipeline for {emiten} Failed ---")

    logging.info(f"--- Succesfully Prepare the Forecasting Data for {len(list_of_emitens) - total_failed_stocks} out of {len(list_of_emitens)} Stocks ---")
    return all_emiten_data