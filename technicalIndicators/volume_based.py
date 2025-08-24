import numpy as np
import pandas as pd
from stock_indicators import indicators

from technicalIndicators.helpers import identify_historical_trends

def calculate_on_balance_volume(prepared_data):
    result = indicators.get_obv(prepared_data, 10)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'On Balance Volume': [val.obv for val in result],
    })

    result_df.dropna(subset=['On Balance Volume'], inplace=True)

    result_df['On Balance Volume Increasing'] = identify_historical_trends(result_df, 'On Balance Volume', 5, make_bool_up=True)
    result_df['On Balance Volume Decreasing'] = identify_historical_trends(result_df, 'On Balance Volume', 5, make_bool_down=True)

    result_df.drop(columns=['On Balance Volume'], inplace=True)

    return result_df.set_index('Date')

def calculate_accumulation_distribution_line(prepared_data):
    result = indicators.get_adl(prepared_data, 10)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Accumulation Distribution Line': [val.adl for val in result],
    })

    result_df.dropna(subset=['Accumulation Distribution Line'], inplace=True)

    result_df['Accumulation Distribution Line Increasing'] = identify_historical_trends(result_df, 'Accumulation Distribution Line', 5, make_bool_up=True)
    result_df['Accumulation Distribution Line Decreasing'] = identify_historical_trends(result_df, 'Accumulation Distribution Line', 5, make_bool_down=True)

    result_df.drop(columns=['Accumulation Distribution Line'], inplace=True)

    return result_df.set_index('Date')

def calculate_chaikin_money_flow(prepared_data):
    result = indicators.get_cmf(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Chaikin Money Flow': [val.cmf for val in result]
    })

    result_df.dropna(subset=['Chaikin Money Flow'], inplace=True)

    result_df['Positive CMF'] = (result_df['Chaikin Money Flow'] > 0).astype(int)
    result_df['Strong Positive CMF'] = (result_df['Chaikin Money Flow'] >= 30).astype(int)
    result_df['Negative CMF'] = (result_df['Chaikin Money Flow'] <= 0).astype(int)
    result_df['Strong Negative CMF'] = (result_df['Chaikin Money Flow'] <= -30).astype(int)
    result_df['Crossover CMF'] = [np.nan] + ((result_df['Chaikin Money Flow'].values[:-1] * result_df['Chaikin Money Flow'].values[1:]) <= 0).astype(int).tolist()

    result_df.drop(columns=['Chaikin Money Flow'], inplace=True)

    return result_df.set_index('Date')

def calculate_money_flow_index(prepared_data):
    result = indicators.get_mfi(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Money Flow Index': [val.mfi for val in result]
    })

    result_df.dropna(subset=['Money Flow Index'], inplace=True)

    result_df['MFI Overbought'] = (result_df['Money Flow Index'] >= 80).astype(int)
    result_df['MFI Oversold'] = (result_df['Money Flow Index'] <= 20).astype(int)
    result_df['MFI Increasing'] = identify_historical_trends(result_df, 'Money Flow Index', 5, make_bool_up=True)
    result_df['MFI Decreasing'] = identify_historical_trends(result_df, 'Money Flow Index', 5, make_bool_down=True)

    result_df.drop(columns=['Money Flow Index'], inplace=True)

    return result_df.set_index('Date')