import pandas as pd
from stock_indicators import indicators

def calculate_bollinger_bands(prepared_data):
    result = indicators.get_bollinger_bands(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Percentage Bollinger': [val.percent_b for val in result],
        'Z Score Bollinger': [val.z_score for val in result],    
        'Width Bollinger': [val.width for val in result]
    })

    return result_df.set_index('Date')

def calculate_keltner(prepared_data):
    result = indicators.get_keltner(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Width Keltner': [val.width for val in result]
    })

    return result_df.set_index('Date')

def calculate_donchian(prepared_data):
    result = indicators.get_donchian(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Width Donchian': [val.width for val in result]
    })

    return result_df.set_index('Date')