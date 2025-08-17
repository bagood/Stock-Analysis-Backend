import pandas as pd
from stock_indicators import indicators

def calculate_on_balance_volume(prepared_data):
    result = indicators.get_obv(prepared_data, 10)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'On Balance Volume': [val.obv for val in result],
        'On Balance Volume Moving Average': [val.obv_sma for val in result]
    })

    return result_df.set_index('Date')

def calculate_accumulation_distribution_line(prepared_data):
    result = indicators.get_adl(prepared_data, 10)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Money Flow Multiplier ADL': [val.money_flow_multiplier for val in result],
        'Money Flow Volume ADL': [val.money_flow_volume for val in result],
        'Accumulation Distribution Line': [val.adl for val in result],
        'Accumulation Distribution Line Moving Average': [val.adl_sma for val in result],
    })

    return result_df.set_index('Date')

def calculate_chaikin_money_flow(prepared_data):
    result = indicators.get_cmf(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Money Flow Multiplier CMF': [val.money_flow_multiplier for val in result],
        'Money Flow Volume CMF': [val.money_flow_volume for val in result],
        'Chaikin Money Flow': [val.cmf for val in result]
    })

    return result_df.set_index('Date')

def calculate_money_flow_index(prepared_data):
    result = indicators.get_mfi(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Money Flow Index': [val.mfi for val in result]
    })

    return result_df.set_index('Date')