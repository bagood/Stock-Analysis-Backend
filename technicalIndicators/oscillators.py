import pandas as pd
from stock_indicators import indicators

def calculate_relative_strength_index(prepared_data):
    result = indicators.get_rsi(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Relative Strength Index': [val.rsi for val in result]
    })
    
    result_df.dropna(subset=['Relative Strength Index'], inplace=True)
    
    result_df['RSI Up Trend'] = (result_df['Relative Strength Index'] >= 65).astype(int)
    result_df['RSI Down Trend'] = (result_df['Relative Strength Index'] <= 35).astype(int)

    result_df.drop(columns=['Relative Strength Index'], inplace=True)

    return result_df.set_index('Date')

def calculate_stochastic_oscillator(prepared_data):
    result = indicators.get_stoch(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Stochastic Oscillator': [val.oscillator for val in result]
    })

    result_df.dropna(subset=['Stochastic Oscillator'], inplace=True)

    result_df['Stochastic Oscillator Overbought'] = (result_df['Stochastic Oscillator'] >= 80).astype(float)
    result_df['Stochastic Oscillator Oversold'] = (result_df['Stochastic Oscillator'] <= 20).astype(float)

    result_df.drop(columns=['Stochastic Oscillator'], inplace=True)

    return result_df.set_index('Date')