import pandas as pd
from stock_indicators import indicators

def calculate_ehler_fisher_transform(prepared_data):
    result = indicators.get_fisher_transform(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Fisher Transform': [val.fisher for val in result],
        'Fisher Transform Trigger': [val.trigger for val in result]
    })

    return result_df.set_index('Date')

def calculate_zig_zag(prepared_data):
    result = indicators.get_zig_zag(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Zig Zag': [val.zig_zag for val in result],
        'Zig Zag Endpoint': [val.point_type for val in result],
        'Retrace High': [val.retrace_high for val in result],
        'Retrace Low': [val.retrace_low for val in result],
    })

    result_df['Zig Zag High'] = result_df['Zig Zag Endpoint'].apply(lambda row: 1 if row == 'H' else 0)
    result_df['Zig Zag Low'] = result_df['Zig Zag Endpoint'].apply(lambda row: 1 if row == 'L' else 0)
    result_df.drop(columns=['Zig Zag Endpoint'], inplace=True)

    return result_df.set_index('Date')