import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the main functions from the data preparation and model development modules.
from modelDevelopment.main import develop_model
from dataPreparation.main import download_stock_data, prepare_data_for_modelling, prepare_data_for_forecasting

# --- Logging Configuration ---
# Configure the logger to provide clear, timestamped updates on the script's progress.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def select_kode_to_model(n_kode_saham_limit: int = 0):
    """
    Selects the most actively traded stocks from a master list.

    This function reads a list of stock tickers from an Excel file, downloads
    their trading volume over the last 45 days, and filters for the top 50%
    most liquid stocks based on average daily volume. This ensures that models
    are built only for stocks with sufficient trading activity.

    Args:
        n_kode_saham_limit (int, optional): The number of stocks to process from the
                                          top of the Excel list. If 0, all stocks
                                          are considered. Defaults to 0.

    Returns:
        np.array: An array of selected stock ticker strings (e.g., ['BBCA', 'TLKM']).
    """
    logging.info("Starting stock selection process based on recent trading volume.")
    
    # Load the full list of stocks or a limited subset for faster testing.
    try:
        if n_kode_saham_limit == 0:
            data_saham = pd.read_excel('performStockAnalysis/daftar_saham.xlsx')
            logging.info("Loaded the full list of stocks from 'daftar_saham.xlsx'.")
        else:
            data_saham = pd.read_excel('performStockAnalysis/daftar_saham.xlsx').head(n_kode_saham_limit)
            logging.info(f"Loaded a limited list of {n_kode_saham_limit} stocks.")
    except FileNotFoundError:
        logging.error("The file 'performStockAnalysis/daftar_saham.xlsx' was not found. Aborting.")
        return np.array([])

    # Define the time window for calculating average volume (last 45 days).
    start_date = (datetime.now().date() - timedelta(days=45)).strftime('%Y-%m-%d')
    all_average_volume = pd.DataFrame()
    
    logging.info(f"Fetching volume data for {len(data_saham)} stocks from {start_date} to today.")
    
    # Loop through each stock ticker to download its recent volume data.
    for i, kode in enumerate(data_saham['Kode'].values):
        logging.info(f"({i+1}/{len(data_saham)}) Fetching volume for ticker: {kode}")
        # Append '.JK' for Jakarta Stock Exchange tickers.
        temp_data = download_stock_data(f'{kode}', start_date, '')
        
        if temp_data is not None and not temp_data.empty:
            # Calculate the mean volume and store it.
            average_volume = temp_data['Volume'].mean()
            all_average_volume = pd.concat((
                all_average_volume,
                pd.DataFrame({'Kode': [kode], 'Average Volume': [average_volume]})
            ), ignore_index=True)
        else:
            logging.warning(f"Could not retrieve data for {kode}. Skipping.")

    if all_average_volume.empty:
        logging.error("No volume data could be fetched for any stock. Aborting selection.")
        return np.array([])
            
    # Determine the number of stocks to select (top 50% rounded up).
    n_selected_kode = int(np.ceil(len(all_average_volume) * 0.5))
    logging.info(f"Calculated selection threshold. Selecting top {n_selected_kode} most traded stocks.")
    
    # Sort stocks by average volume in descending order and select the top N.
    selected_kode = all_average_volume.sort_values('Average Volume', ascending=False) \
                                      .head(n_selected_kode)['Kode'].values
    
    logging.info(f"Stock selection complete. Selected tickers: {list(selected_kode)}")
    return selected_kode

def _combine_train_test_metrics_into_single_df(kode: str, train_metrics: dict, test_metrics: dict):
    """
    (Internal Helper) Combines training and testing metrics into a single DataFrame row.

    Args:
        kode (str): The stock ticker symbol.
        train_metrics (dict): A dictionary of performance metrics for the training set.
        test_metrics (dict): A dictionary of performance metrics for the testing set.

    Returns:
        pd.DataFrame: A single-row DataFrame containing all metrics, prefixed with
                      'Train - ' or 'Test - ', and the stock 'Kode'.
    """
    # Convert the training metrics dictionary to a DataFrame and prefix columns.
    train_df = pd.DataFrame(train_metrics)
    train_df.columns = [f'Train - {col}' for col in train_df.columns]

    # Convert the testing metrics dictionary to a DataFrame and prefix columns.
    test_df = pd.DataFrame(test_metrics)
    test_df.columns = [f'Test - {col}' for col in test_df.columns]

    # Concatenate the two DataFrames side-by-side (axis=1).
    train_test_df = pd.concat((train_df, test_df), axis=1)
    
    # Add the stock ticker as the identifier for the row.
    train_test_df.insert(0, 'Kode', kode)

    return train_test_df
    
def save_developed_model(model, kode: str, model_type: str):
    """
    Saves a trained model object to a file using pickle.

    The filename is standardized to include the stock ticker, model type (e.g., '10dd'),
    and the date of creation.

    Args:
        model (object): The trained model object to be saved.
        kode (str): The stock ticker symbol.
        model_type (str): A descriptor for the model type (e.g., '10dd', '15dd').
    """
    # Create a standardized filename.
    developed_date = datetime.now().date().strftime('%Y%m%d')
    filename = f'database/developedModels/{kode}-{model_type}-{developed_date}.pkl'
    
    logging.info(f"Saving model to '{filename}'...")
    try:
        # Use 'wb' (write binary) mode, which is required for pickling.
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        logging.info("Model saved successfully.")
    except IOError as e:
        logging.error(f"Failed to save model to '{filename}'. Error: {e}")
    
    return

def develop_models_for_selected_kode(selected_kode: list):
    """
    Orchestrates the model development pipeline for a list of selected stocks.

    For each stock ticker, this function will:
    1. Prepare the data by generating features and target variables.
    2. Develop two distinct models (for 10-day and 15-day future trends).
    3. Save each trained model to a file.
    4. Aggregate the performance metrics of all models into summary DataFrames.

    Args:
        selected_kode (list): A list of stock ticker symbols to process.

    """
    logging.info(f"--- Starting Bulk Model Development for {len(selected_kode)} Selected Stocks ---")
    
    # Initialize empty DataFrames to store performance metrics from all runs.
    all_model_performances_10_days = pd.DataFrame()
    all_model_performances_15_days = pd.DataFrame()
    
    # Store all stock that failed to be processed
    failed_stocks = []

    # Loop through each of the selected stock tickers.
    for i, kode in enumerate(selected_kode):
        try:
            logging.info(f"--- Processing Ticker: {kode} ({i+1}/{len(selected_kode)}) ---")
            
            # --- Step 1: Data Preparation ---
            logging.info(f"Preparing data for {kode} with 10 and 15-day trend targets...")
            prepared_data = prepare_data_for_modelling(
                emiten=kode, 
                start_date='2021-01-01', 
                end_date='', 
                target_column='Close', 
                rolling_windows=[10, 15], 
                download=True
            )

            if prepared_data.empty:
                logging.warning(f"Data preparation for {kode} resulted in an empty DataFrame. Skipping model development.")
                continue
            
            # --- Step 2: Model Development ---
            logging.info(f"Developing model for '{kode}' - 10 Day Trend...")
            model_10_days, train_metrics_10_days, test_metrics_10_days = develop_model(prepared_data, 'Upcoming 10 Days Trend')
            
            logging.info(f"Developing model for '{kode}' - 15 Day Trend...")
            model_15_days, train_metrics_15_days, test_metrics_15_days = develop_model(prepared_data, 'Upcoming 15 Days Trend')

            # --- Step 3: Save Models and Aggregate Performance ---
            logging.info(f"Saving models and collating performance metrics for {kode}...")
            save_developed_model(model_10_days, kode, '10dd')
            save_developed_model(model_15_days, kode, '15dd')

            # Combine the train/test metrics for each model into a clean DataFrame row.
            train_test_10_days = _combine_train_test_metrics_into_single_df(kode, train_metrics_10_days, test_metrics_10_days)
            train_test_15_days = _combine_train_test_metrics_into_single_df(kode, train_metrics_15_days, test_metrics_15_days)

            # Append the performance row to the aggregate DataFrames.
            all_model_performances_10_days = pd.concat((all_model_performances_10_days, train_test_10_days), ignore_index=True)
            all_model_performances_15_days = pd.concat((all_model_performances_15_days, train_test_15_days), ignore_index=True)
            
            logging.info(f"--- Finished processing for Ticker: {kode} ---")
        
        except:
            failed_stocks.append(kode)
            logging.warning(f"--- Failed processing for Ticker: {kode} ---")

    # Saves the developed model performances to the database
    developed_date = datetime.now().date().strftime('%Y%m%d')
    filename_10dd = f'database/modelPerformances/modelPerformance-10dd-{developed_date}.csv'
    filename_15dd = f'database/modelPerformances/modelPerformance-15dd-{developed_date}.csv'

    all_model_performances_10_days.to_csv(filename_10dd, index=False)
    logging.info(f"Succefully saved model performances on forecasting upcoming 10 days trend to {filename_10dd}.")

    all_model_performances_15_days.to_csv(filename_15dd, index=False)
    logging.info(f"Succefully saved model performances on forecasting upcoming 15 days trend to {filename_15dd}.")

    # Take notes on all stocks that gets failed being processed
    try:
        failed_stock_path = f'database/modelPerformances/failedStocks-{developed_date}.txt'
        logging.info(f"Saving stocks that are failed being processed to '{failed_stock_path }'...")
        with open(failed_stock_path, "w") as file:
            for failed_stock in failed_stocks:
                file.write(failed_stock + "\n")
        logging.info("List of failed stocks saved successfully.")
    except IOError as e:
        logging.error(f"Failed to write list of failed stocks to file: {e}")

    logging.info("--- Bulk Model Development Complete. ---")
    return

def make_forecasts_using_the_developed_models(forecast_dd: int, development_date: str, min_test_gini: float):
    """
    Orchestrates the forecasting pipeline for stocks that exceeds the minimum test gini performance.
    """
    logging.info(f"--- Starting the Process of {forecast_dd} Days Forecasting ---")
    
    # Loads the models and sort them by test gini
    model_performance_dd_path = f'database/modelPerformances/modelPerformance-{forecast_dd}dd-{development_date}.csv'
    logging.info(f"Loading the data from {model_performance_dd_path}.")
    all_model_performances_days = pd.read_csv(model_performance_dd_path) \
                                            .sort_values('Test - Gini', ascending=False)
    
    # Select all models that exceeds the minimum test gini performances
    selected_model_performances_days = all_model_performances_days[all_model_performances_days['Test - Gini'] >= min_test_gini] \
                                            .reset_index(drop=True)
    selected_kode = selected_model_performances_days['Kode'].unique()
    logging.info(f"Selected a total of {len(selected_kode)} stocks that exceed the minimum model performance.")
    
    # Prepare the forecasting data
    forecasting_data = prepare_data_for_forecasting(
        list_of_emitens=selected_kode, 
        start_date='2021-01-01', 
        end_date='', 
        rolling_window=forecast_dd, 
        download=True
    )

    # Load the feature columns
    feature_file = 'modelDevelopment/technical_indicator_features.txt'
    try:
        logging.info(f"Loading feature names from '{feature_file}'.")
        with open(feature_file, "r") as file:
            feature_columns = [line.strip() for line in file]
        logging.info(f"Loaded {len(feature_columns)} features.")
    except FileNotFoundError:
        logging.error(f"Feature file not found at '{feature_file}'. Aborting.")
        return

    # Load the developed models
    logging.info(f"Loading the developed models for {len(selected_kode)} stocks")
    model_store = {}
    for kode in selected_kode:
        model_path = f'database/developedModels/{kode}-{forecast_dd}dd-{development_date}.pkl'         
        try:
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            model_store[kode] = loaded_model
        except FileNotFoundError:
            logging.info(f"Error: The file '{model_path}' was not found.")
        except Exception as e:
            logging.info(f"An error occurred while loading the object: {e}")
    logging.info(f"Sucessfully load {len(model_store.keys())} out of {len(selected_kode)} models.")

    logging.info('Starting the forecasting using the loaded models on the loaded forecasting data.')
    # Iterate over each row, get the model according to the kode, and make predictions
    forecasting_data[f'Forecast {forecast_dd} Up Days Trend'] = forecasting_data.apply(
        lambda row: model_store[row['Kode']].predict_proba(row[feature_columns].values.reshape(1, -1))[:, np.where(model_store[row['Kode']].classes_ == 'Up Trend')[0][0]][0]
                    if row['Kode'] in model_store else np.nan,
        axis=1
    )

    forecast_path = f'database/forecastedStocks/forecast-{forecast_dd}dd-{development_date}.csv'
    logging.info(f'Saving the forecast data to {forecast_path}')

    # Save the selected columns of the forecasting data
    selected_columns = ['Kode', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', f'Forecast {forecast_dd} Up Days Trend']
    forecasting_data[selected_columns].to_csv(forecast_path, index=False)

    logging.info(f"--- Finished the Process of {forecast_dd} Days Forecasting ---")
    return