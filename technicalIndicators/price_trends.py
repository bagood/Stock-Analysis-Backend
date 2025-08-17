import pandas as pd
from stock_indicators import indicators

def calculate_atr_trailing_stop(prepared_data):
    result = indicators.get_atr_stop(prepared_data)
    result_data = pd.DataFrame({
        'Date': [val.date for val in result],
        'ATR Stop': [val.atr_stop for val in result],
        'Buy Stop': [val.buy_stop for val in result],
        'Sell Stop': [val.sell_stop for val in result]
    })

    return result_data.set_index('Date')

def calculate_aroon(prepared_data):
    result = indicators.get_aroon(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Aroon Up': [val.aroon_up for val in result],
        'Aroon Down': [val.aroon_down for val in result],
        'Oscillator': [val.oscillator for val in result]
    })

    return result_df.set_index('Date')

def calculate_average_directional_index(prepared_data):
    result = indicators.get_adx(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Plus Directional Index': [val.pdi for val in result],
        'Minus Directional Index': [val.mdi for val in result],
        'Average Directional Index': [val.adx for val in result],
        'Average Directional Index Rating': [val.adxr for val in result]
    })

    return result_df.set_index('Date')

def calculate_elder_ray_index(prepared_data):
    result = indicators.get_elder_ray(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Bull Power': [val.bull_power for val in result],
        'Bear Power': [val.bear_power for val in result]
    })

    return result_df.set_index('Date')

def calculate_moving_average_convergence_divergence(prepared_data):
    result = indicators.get_macd(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Histogram MACD': [val.histogram for val in result]
        
    })

    return result_df.set_index('Date')