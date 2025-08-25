import numpy as np
import pandas as pd
from stock_indicators import indicators

from technicalIndicators.helper import identify_historical_trends

def calculate_ehler_fisher_transform(prepared_data):
    result = indicators.get_fisher_transform(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Fisher Transform': [val.fisher for val in result],
        'Fisher Transform Trigger': [val.trigger for val in result]
    })

    result_df.dropna(subset=['Fisher Transform', 'Fisher Transform Trigger'], inplace=True)

    result_df['Fisher Up trend'] = (result_df['Fisher Transform'] >= 2).astype(int)
    result_df['Fisher Down Trend'] = (result_df['Fisher Transform'] <= -2).astype(int)
    
    previous_position = result_df['Fisher Transform'].values >= result_df['Fisher Transform Trigger'].values
    result_df['Fisher Reversal'] = [np.nan] + (previous_position[:-1] != previous_position[1:]).astype(int).tolist()

    result_df.drop(columns=['Fisher Transform', 'Fisher Transform Trigger'], inplace=True)

    return result_df.set_index('Date')

def calculate_zig_zag(prepared_data):
    result = indicators.get_zig_zag(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Zig Zag': np.array([val.zig_zag for val in result]).astype(float),
        'Zig Zag Endpoint': [val.point_type for val in result]
    })

    result_df.dropna(subset=['Zig Zag'], inplace=True)

    result_df['Zig Zag High'] = result_df['Zig Zag Endpoint'].apply(lambda row: 1 if row == 'H' else 0)
    result_df['Zig Zag Low'] = result_df['Zig Zag Endpoint'].apply(lambda row: 1 if row == 'L' else 0)
    result_df['Zig Zag Increasing'] = identify_historical_trends(result_df, 'Zig Zag', 5, make_bool_up=True)
    result_df['Zig Zag Decreasing'] = identify_historical_trends(result_df, 'Zig Zag', 5, make_bool_down=True)
    
    result_df.drop(columns=['Zig Zag', 'Zig Zag Endpoint'], inplace=True)

    return result_df.set_index('Date')