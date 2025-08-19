import pickle
import pandas as pd
from datetime import datetime

def _combine_train_test_metrics_into_single_df(kode: str, train_metrics: dict, test_metrics: dict) -> pd.DataFrame:
    """
    (Internal Helper) Combines training and testing metrics into a single DataFrame row.

    Args:
        kode (str): The stock ticker symbol.
        train_metrics (dict): A dictionary of performance metrics for the training set.
        test_metrics (dict): A dictionary of performance metrics for the testing set.

    Returns:
        pd.DataFrame: A single-row DataFrame containing all metrics, prefixed with
                      'Train - ' or 'Test - ', and the stock 'Kode'.
    """
    train_df = pd.DataFrame(train_metrics)
    train_df.columns = [f'Train - {col}' for col in train_df.columns]

    test_df = pd.DataFrame(test_metrics)
    test_df.columns = [f'Test - {col}' for col in test_df.columns]

    train_test_df = pd.concat((train_df, test_df), axis=1)
    train_test_df.insert(0, 'Kode', kode)

    return train_test_df

def _save_developed_model(model, kode: str, model_type: str):
    """
    Saves a trained model object to a file using pickle.

    The filename is standardized to include the stock ticker, model type (e.g., '10dd'),
    and the date of creation.

    Args:
        model (object): The trained model object to be saved.
        kode (str): The stock ticker symbol.
        model_type (str): A descriptor for the model type (e.g., '10dd', '15dd').
    """ 
    developed_date = datetime.now().date().strftime('%Y%m%d')
    filename = f'database/developedModels/{kode}-{model_type}-{developed_date}.pkl'
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    return