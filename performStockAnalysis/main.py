import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from modelDevelopment.main import develop_model
from dataPreparation.helper import _download_stock_data
from dataPreparation.main import prepare_data_for_modelling, prepare_data_for_forecasting
from performStockAnalysis.helper import _combine_train_test_metrics_into_single_df, _save_developed_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def select_kode_to_model(n_kode_saham_limit: int = 0) -> np.array:
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
        np.array: An array of selected stock ticker strings
    """
    logging.info("Starting stock selection process based on recent trading volume")
    
    if n_kode_saham_limit == 0:
        data_saham = pd.read_excel('performStockAnalysis/daftar_saham.xlsx')
        logging.info("Loaded the full list of stocks from 'daftar_saham.xlsx'")
    else:
        data_saham = pd.read_excel('performStockAnalysis/daftar_saham.xlsx').head(n_kode_saham_limit)
        logging.info(f"Loaded a limited list of {n_kode_saham_limit} stocks")

    start_date = (datetime.now().date() - timedelta(days=45)).strftime('%Y-%m-%d')
    all_average_volume = pd.DataFrame()
    logging.info(f"Fetching volume data for {len(data_saham)} stocks from {start_date} to today")

    for i, kode in enumerate(data_saham['Kode'].values):
        logging.info(f"({i+1}/{len(data_saham)}) Fetching volume for ticker: {kode}")
        temp_data = _download_stock_data(f'{kode}', start_date, '')
        all_average_volume = pd.concat((
            all_average_volume,
            pd.DataFrame({
                'Kode': [kode], 
                'Average Volume': [temp_data['Volume'].mean()]
            })
        ), ignore_index=True)
            
    n_selected_kode = int(np.ceil(len(all_average_volume) * 0.5))
    logging.info(f"Calculated selection threshold. Selecting top {n_selected_kode} most traded stocks")
    
    selected_kode = all_average_volume.sort_values('Average Volume', ascending=False) \
                                      .head(n_selected_kode)['Kode'].values
    logging.info(f"Stock selection complete. Selected tickers: {list(selected_kode)}")

    return selected_kode

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
    logging.info(f"Starting Bulk Model Development for {len(selected_kode)} Selected Stocks")
    
    all_model_performances_10_days = pd.DataFrame()
    all_model_performances_15_days = pd.DataFrame()
    failed_stocks = []

    for i, kode in enumerate(selected_kode):
        logging.info(f"Processing Ticker: {kode} ({i+1}/{len(selected_kode)})")
        
        logging.info(f"Preparing data for {kode} with 10 and 15-day trend targets")
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
        
        logging.info(f"Developing model for '{kode}' - 10 Day Trend")
        model_10_days, train_metrics_10_days, test_metrics_10_days = develop_model(prepared_data, 'Upcoming 10 Days Trend')
        
        logging.info(f"Developing model for '{kode}' - 15 Day Trend")
        model_15_days, train_metrics_15_days, test_metrics_15_days = develop_model(prepared_data, 'Upcoming 15 Days Trend')

        logging.info(f"Saving models and collating performance metrics for {kode}")
        _save_developed_model(model_10_days, kode, '10dd')
        _save_developed_model(model_15_days, kode, '15dd')

        logging.info(f"Measuring model performances on training and testing sets")
        train_test_10_days = _combine_train_test_metrics_into_single_df(kode, train_metrics_10_days, test_metrics_10_days)
        train_test_15_days = _combine_train_test_metrics_into_single_df(kode, train_metrics_15_days, test_metrics_15_days)
        all_model_performances_10_days = pd.concat((all_model_performances_10_days, train_test_10_days), ignore_index=True)
        all_model_performances_15_days = pd.concat((all_model_performances_15_days, train_test_15_days), ignore_index=True)
        
        logging.info(f"Finished processing for Ticker: {kode}")

    developed_date = datetime.now().date().strftime('%Y%m%d')
    filename_10dd = f'database/modelPerformances/modelPerformance-10dd-{developed_date}.csv'
    filename_15dd = f'database/modelPerformances/modelPerformance-15dd-{developed_date}.csv'

    all_model_performances_10_days.to_csv(filename_10dd, index=False)
    logging.info(f"Succefully saved model performances on forecasting upcoming 10 days trend to {filename_10dd}")

    all_model_performances_15_days.to_csv(filename_15dd, index=False)
    logging.info(f"Succefully saved model performances on forecasting upcoming 15 days trend to {filename_15dd}")

    failed_stock_path = f'database/modelPerformances/failedStocks-{developed_date}.txt'
    logging.info(f"Saving stocks that are failed being processed to '{failed_stock_path }'...")
    with open(failed_stock_path, "w") as file:
        for failed_stock in failed_stocks:
            file.write(failed_stock + "\n")
    logging.info("List of failed stocks saved successfully")

    logging.info("Bulk Model Development Complete")

    return

def make_forecasts_using_the_developed_models(forecast_dd: int, development_date: str, min_test_gini: float):
    """
    Orchestrates the forecasting pipeline for stocks that exceeds the minimum test gini performance.
    
    Args:
        forecast_dd (int): The desired upcoming days to be forcasted
        development_date (str): The date where the model is developed
        min_test_gini (float): The minimum gini performance for the model's testing performance
    """
    logging.info(f"Starting the Process of {forecast_dd} Days Forecasting")
    
    model_performance_dd_path = f'database/modelPerformances/modelPerformance-{forecast_dd}dd-{development_date}.csv'
    logging.info(f"Loading the data from {model_performance_dd_path}")
    all_model_performances_days = pd.read_csv(model_performance_dd_path) \
                                            .sort_values('Test - Gini', ascending=False)
    
    logging.info(f'Select stock with the model performance on testing data that is greater than {min_test_gini}')
    selected_model_performances_days = all_model_performances_days[all_model_performances_days['Test - Gini'] >= min_test_gini] \
                                            .reset_index(drop=True)
    selected_kode = selected_model_performances_days['Kode'].unique()
    logging.info(f"Selected a total of {len(selected_kode)} stocks that exceed the minimum model performance")
    
    logging.info('Prepare all stock data to be used for forecasting')
    forecasting_data = prepare_data_for_forecasting(
        list_of_emitens=selected_kode, 
        start_date='2021-01-01', 
        end_date='', 
        rolling_window=forecast_dd, 
        download=True
    )

    feature_file = 'modelDevelopment/technical_indicator_features.txt'
    logging.info(f"Loading feature names from '{feature_file}'")
    with open(feature_file, "r") as file:
        feature_columns = [line.strip() for line in file]
    logging.info(f"Loaded {len(feature_columns)} features")

    logging.info(f"Loading the developed models for {len(selected_kode)} stocks")
    model_store = {}
    for kode in selected_kode:
        model_path = f'database/developedModels/{kode}-{forecast_dd}dd-{development_date}.pkl'         
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        model_store[kode] = loaded_model
    logging.info(f"Sucessfully load {len(model_store.keys())} out of {len(selected_kode)} models")

    logging.info('Starting the forecasting using the loaded models on the prepared forecasting data')
    forecasting_data[f'Forecast - Upcoming {forecast_dd} Days Trend'] = forecasting_data.apply(
        lambda row: model_store[row['Kode']].predict(row[feature_columns].values.reshape(1, -1))[0] if row['Kode'] in model_store else np.nan,
        axis=1
    )

    forecast_path = f'database/forecastedStocks/forecast-{forecast_dd}dd-{development_date}.csv'
    logging.info(f'Saving the forecast data to {forecast_path}')

    selected_columns = ['Kode', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', f'Forecast - Upcoming {forecast_dd} Days Trend']
    forecasting_data.loc[forecasting_data['Date'] == forecasting_data['Date'].max(), selected_columns] \
                        .to_csv(forecast_path, index=False)

    logging.info(f"Finished the Process of {forecast_dd} Days Forecasting")

    return