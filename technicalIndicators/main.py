import logging
import pandas as pd
from stock_indicators import Quote

from technicalIndicators.price_trends import calculate_atr_trailing_stop, calculate_aroon, calculate_average_directional_index, calculate_elder_ray_index, calculate_moving_average_convergence_divergence
from technicalIndicators.price_channels import calculate_keltner, calculate_donchian, calculate_bollinger_bands
from technicalIndicators.oscillators import calculate_relative_strength_index, calculate_stochastic_oscillator
from technicalIndicators.volume_based import calculate_on_balance_volume, calculate_money_flow_index, calculate_chaikin_money_flow, calculate_accumulation_distribution_line
from technicalIndicators.price_transformations import calculate_ehler_fisher_transform, calculate_zig_zag

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def _prepare_data_for_generating_stock_indicators(data: pd.DataFrame) -> list:
    """
    Converts pandas dataframe into a type that is suitable for generating technical indicators using stock_indicators.

    Args:
        data (pd.DataFrame): Stock data containing the column Date, Open, High, Low, Close, and Volume

    Returns:
        list: Converted pandas dataframe ready to be used for generating the technical indicators
    """
    data['Date'] = pd.to_datetime(data['Date'])
    prepared_data = [
        Quote(d,o,h,l,c,v) 
        for d,o,h,l,c,v 
        in zip(data['Date'], data['Open'], data['High'], data['Low'], data['Close'], data['Volume'])
    ]
    
    return prepared_data

def generate_all_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all the technical indicators, each having different unique characteristics of the stock to be measures

    Each of the stocks characteristics to be measured are as follows:
    1. Price Trends
    2. Price Channels
    3. Oscillators
    4. Volume Based
    5. Price Transformations

    Args:
        data (pd.DataFrame): Stock data containing the column Date, Open, High, Low, Close, and Volume

    Returns:
        list: Converted pandas dataframe ready to be used for generating the technical indicators
    """
    logging.info('Processing the data into a format that is compatible for generating the stock indicators')
    prepared_data = _prepare_data_for_generating_stock_indicators(data) 

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    original_columns = set(data.columns)

    logging.info('Generating stock indicators of price trends')
    price_trends = [
        calculate_atr_trailing_stop(prepared_data),
        calculate_aroon(prepared_data),
        calculate_average_directional_index(prepared_data), 
        calculate_elder_ray_index(prepared_data), 
        calculate_moving_average_convergence_divergence(prepared_data)
    ]

    logging.info('Generating stock indicators of price channels')
    price_channels = [
        calculate_keltner(prepared_data), 
        calculate_donchian(prepared_data),
        calculate_bollinger_bands(prepared_data)
    ]

    logging.info('Generating stock indicators of oscillators')
    oscillators = [
        calculate_relative_strength_index(prepared_data), 
        calculate_stochastic_oscillator(prepared_data)
    ]

    logging.info('Generating stock indicators of volume based')
    volume_based = [
        calculate_on_balance_volume(prepared_data), 
        calculate_money_flow_index(prepared_data), 
        calculate_chaikin_money_flow(prepared_data),
        calculate_accumulation_distribution_line(prepared_data)
    ]

    logging.info('Generating stock indicators of price transformations')
    price_transformations = [
        calculate_ehler_fisher_transform(prepared_data), 
        calculate_zig_zag(prepared_data)
    ]

    all_stock_indicators = price_trends + price_channels + oscillators + volume_based + price_transformations
    all_stock_indicators_data = data.join(all_stock_indicators)
    logging.info('Successfullu generated all stock indicators')

    updated_columns = set(all_stock_indicators_data.columns)
    feature_columns = sorted(list(updated_columns - original_columns))
    logging.info(f'Acquired {len(feature_columns)} features for the stock data')

    output_path = 'modelDevelopment/technical_indicator_features.txt'
    logging.info(f'Saving all newly generated features into {output_path}')
    with open(output_path, "w") as file:
        for fea_col in feature_columns:
            file.write(fea_col + "\n")

    return all_stock_indicators_data