import pandas as pd
from stock_indicators import indicators

def calculate_relative_strength_index(prepared_data):
    result = indicators.get_rsi(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Relative Strength Index': [val.rsi for val in result]
    })

    return result_df.set_index('Date')

def calculate_stochastic_oscillator(prepared_data):
    result = indicators.get_stoch(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Stochastic Oscilator': [val.oscillator for val in result],
        'Signal Stochastic Oscilator': [val.signal for val in result],
        'Percent Stochastic Oscilator': [val.percent_j for val in result],
    })

    return result_df.set_index('Date')