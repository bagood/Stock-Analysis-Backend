import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from dataPreparation.main import download_stock_data, prepare_data_for_modelling
from modelDevelopment.main import develop_model

def select_kode_to_model(n_kode_saham_limit=0):
    if n_kode_saham_limit == 0:
        data_saham = pd.read_excel('performStockAnalysis/Daftar Saham.xlsx')
    else:
        data_saham = pd.read_excel('performStockAnalysis/Daftar Saham.xlsx').head(n_kode_saham_limit)
    
    start_date = (datetime.now().date() - timedelta(days=45)).strftime('%Y-%m-%d')
    all_average_volume = pd.DataFrame()

    for kode in data_saham['Kode'].values:
        temp_data = download_stock_data(f'{kode}.JK', start_date, '')
        average_volume = temp_data['Volume'].mean()
        all_average_volume = pd.concat((
            all_average_volume,
            pd.DataFrame({
                'Kode': [kode],
                'Average Volume': [average_volume]
            })
        ))

    n_selected_kode = int(np.ceil(len(all_average_volume) * 0.5))
    selected_kode = all_average_volume.sort_values('Average Volume', ascending=False) \
                                            .head(n_selected_kode) \
                                            ['Kode'] \
                                            .values

    return selected_kode

def _combine_train_test_metrics_into_single_df(kode, train_metrics, test_metrics):
    train_df = pd.DataFrame(train_metrics)
    train_df.columns = [f'Train - {col}' for col in train_df.columns]

    test_df = pd.DataFrame(test_metrics)
    test_df.columns = [f'Test - {col}' for col in test_df.columns]

    train_test_df = pd.concat((train_df, test_df), axis=1)
    train_test_df.reset_index(inplace=True)
    train_test_df.rename(columns={'index': 'Kode'}, inplace=True)
    train_test_df['Kode'] = kode

    return train_test_df
    
def save_developed_model(model, kode, type):
    developed_date = datetime.now().date().strftime('%Y%m%d')
    filename = f'developedModels/{kode}-{type}-{developed_date}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    return

def develop_models_for_selected_kode(selected_kode):
    all_model_performances_10_days = pd.DataFrame()
    all_model_performances_15_days = pd.DataFrame()
    
    for kode in selected_kode:
        prepared_data = prepare_data_for_modelling(f'{kode}.JK', '2021-01-01', '', 'Close', [10, 15], download=False)

        model_10_days, train_metrics_10_days, test_metrics_10_days = develop_model(prepared_data, 'Upcoming 10 Days Trend')
        model_15_days, train_metrics_15_days, test_metrics_15_days = develop_model(prepared_data, 'Upcoming 15 Days Trend')

        save_developed_model(model_10_days, kode, '10dd')
        save_developed_model(model_15_days, kode, '15dd')

        train_test_10_days = _combine_train_test_metrics_into_single_df(kode, train_metrics_10_days, test_metrics_10_days)
        train_test_15_days = _combine_train_test_metrics_into_single_df(kode, train_metrics_15_days, test_metrics_15_days)

        all_model_performances_10_days = pd.concat((all_model_performances_10_days, train_test_10_days))
        all_model_performances_15_days = pd.concat((all_model_performances_15_days, train_test_15_days))

    return all_model_performances_10_days, all_model_performances_15_days