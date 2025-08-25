import numpy as np
import pandas as pd
from stock_indicators import indicators

from technicalIndicators.helper import identify_historical_trends

def calculate_bollinger_bands(data, prepared_data):
    result = indicators.get_bollinger_bands(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Upper Band': [val.upper_band for val in result],
        'Lower Band': [val.lower_band for val in result],    
        'Width': [val.width for val in result]
    })

    result_df['Bollinger Overbought'] = np.array(result_df['Upper Band'].values <= data['Close'].values).astype(int)
    result_df['Bollinger Oversold'] = np.array(result_df['Lower Band'].values >= data['Close'].values).astype(int)
    result_df['Width Bollinger Increasing'] = identify_historical_trends(result_df, 'Width', 5, make_bool_up=True)
    result_df['Width Bollinger Decreasing'] = identify_historical_trends(result_df, 'Width', 5, make_bool_down=True)
    
    result_df.dropna(subset=['Upper Band', 'Lower Band', 'Width'], inplace=True)
    result_df.drop(columns=['Upper Band', 'Lower Band', 'Width'], inplace=True)

    return result_df.set_index('Date')

def calculate_keltner(data, prepared_data):
    result = indicators.get_keltner(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Upper Band': [val.upper_band for val in result],
        'Lower Band': [val.lower_band for val in result],
        'Width': [val.width for val in result]
    })

    result_df['Keltner Overbought'] = np.array(result_df['Upper Band'].values <= data['Close'].values).astype(int)
    result_df['Keltner Oversold'] = np.array(result_df['Lower Band'].values >= data['Close'].values).astype(int)
    result_df['Width Keltner Increasing'] = identify_historical_trends(result_df, 'Width', 5, make_bool_up=True)
    result_df['Width Keltner Decreasing'] = identify_historical_trends(result_df, 'Width', 5, make_bool_down=True)

    result_df.dropna(subset=['Upper Band', 'Lower Band', 'Width'], inplace=True)
    result_df.drop(columns=['Upper Band', 'Lower Band', 'Width'], inplace=True)

    return result_df.set_index('Date')

def calculate_donchian(data, prepared_data):
    result = indicators.get_donchian(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Upper Band': np.array([np.nan if val.upper_band == None else val.upper_band for val in result]).astype(float),
        'Lower Band': np.array([np.nan if val.lower_band == None else val.lower_band for val in result]).astype(float),
        'Width': np.array([np.nan if val.width == None else val.width for val in result]).astype(float)
    })
    
    result_df['Donchian Overbought'] = np.array(result_df['Upper Band'].values <= data['Close'].values).astype(int)
    result_df['Donchian Oversold'] = np.array(result_df['Lower Band'].values >= data['Close'].values).astype(int)
    result_df['Width Donchian Increasing'] = identify_historical_trends(result_df, 'Width', 5, make_bool_up=True)
    result_df['Width Donchian Decreasing'] = identify_historical_trends(result_df, 'Width', 5, make_bool_down=True)

    result_df.dropna(subset=['Upper Band', 'Lower Band', 'Width'], inplace=True)
    result_df.drop(columns=['Upper Band', 'Lower Band', 'Width'], inplace=True)

    return result_df.set_index('Date')