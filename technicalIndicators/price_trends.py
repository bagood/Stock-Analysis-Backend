import numpy as np
import pandas as pd
from stock_indicators import indicators

from technicalIndicators.helper import identify_historical_trends

def calculate_atr_trailing_stop(data, prepared_data):
    result = indicators.get_atr_stop(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'ATR Stop': [val.atr_stop if val.atr_stop != None else np.nan for val in result]
    })

    result_df['ATR Bullish'] = (result_df['ATR Stop'].values >= data['Close'].values).astype(int)
    result_df['ATR Bearish'] = (result_df['ATR Stop'].values < data['Close'].values).astype(int)
    result_df.dropna(subset='ATR Stop', inplace=True)

    result_df.drop(columns=['ATR Stop'], inplace=True)    

    return result_df.set_index('Date')

def calculate_aroon(prepared_data):
    result = indicators.get_aroon(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Aroon Up': [val.aroon_up for val in result],
        'Aroon Down': [val.aroon_down for val in result]
    })

    result_df.dropna(subset=['Aroon Up'], inplace=True)
    aroon_position = result_df['Aroon Up'].values >= result_df['Aroon Down'].values
    result_df['Aroon Change Position'] = [np.nan] + (aroon_position[:1] != aroon_position[1:]).astype(int).tolist()
    result_df['Aroon Up Trend'] = (result_df['Aroon Up'].values >= 70).astype(int)
    result_df['Aroon Down Trend'] = (result_df['Aroon Down'].values >= 70).astype(int)

    result_df.drop(columns=['Aroon Up', 'Aroon Down'], inplace=True)

    return result_df.set_index('Date')

def calculate_average_directional_index(prepared_data):
    result = indicators.get_adx(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Plus Directional Index': [val.pdi for val in result],  
        'Minus Directional Index': [val.mdi for val in result]
    })

    result_df.dropna(subset=['Plus Directional Index', 'Minus Directional Index'], inplace=True)
    result_df['Weak Plus Directional Index'] = (result_df['Plus Directional Index'] <= 25).astype(int)
    result_df['Strong Plus Directional Index'] = (result_df['Plus Directional Index'] > 25).astype(int)
    result_df['Weak Minus Directional Index'] = (result_df['Minus Directional Index'] <= 25).astype(int)
    result_df['Strong Minus Directional Index'] = (result_df['Minus Directional Index'] > 25).astype(int)

    result_df.drop(columns=['Plus Directional Index', 'Minus Directional Index'], inplace=True)

    return result_df.set_index('Date')

def calculate_elder_ray_index(prepared_data):
    result = indicators.get_elder_ray(prepared_data)

    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Bull Power': [val.bull_power for val in result],
        'Bear Power': [val.bear_power for val in result]
    })

    result_df.dropna(subset=['Bull Power', 'Bear Power'], inplace=True)

    result_df['Bull Power Up Trend'] = identify_historical_trends(result_df, 'Bull Power', 10, make_bool_up=True)
    result_df['Bull Power 80% Positives'] = np.array([np.nan if i < 10 else np.sum(result_df['Bull Power'].values[i-10:i] > 0) >= 8 for i in range(len(result_df))])

    result_df['Bear Power Trend'] = identify_historical_trends(result_df, 'Bear Power', 10, make_bool_down=True)
    result_df['Bear Power 80% Negatives'] = np.array([np.nan if i < 10 else np.sum(result_df['Bear Power'].values[i-10:i] < 0) >= 8 for i in range(len(result_df))])

    result_df.drop(columns=['Bull Power', 'Bear Power'], inplace=True)

    return result_df.set_index('Date')

def calculate_moving_average_convergence_divergence(prepared_data):
    result = indicators.get_macd(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Histogram MACD': [val.histogram for val in result]
        
    })

    result_df.dropna(subset=['Histogram MACD'], inplace=True)

    result_df['MACD Near 0'] = (np.abs(result_df['Histogram MACD']) < 10).astype(int)
    result_df['MACD Non-Near 0 Negative'] = (result_df['Histogram MACD'] <= -10).astype(int)
    result_df['MACD Non-Near 0 Positive'] = (result_df['Histogram MACD'] >= 10).astype(int)

    result_df.drop(columns=['Histogram MACD'], inplace=True)

    return result_df.set_index('Date')