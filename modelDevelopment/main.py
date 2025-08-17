import logging
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from catboost import CatBoostClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def split_data_to_train_val_test(data: pd.DataFrame, feature_columns: list, target_column: str):
    """
    Splits time-series data into training and testing sets.

    This function implements a time-based split crucial for financial forecasting:
    - Test Set: The most recent 30 days of data with valid targets.
    - Training Set: All data preceding the test set.
    - Validation Set (for Hyperparameter Tuning): The last 30 days of the training set.

    Args:
        data (pd.DataFrame): The complete DataFrame containing features and the target.
        feature_columns (list): A list of column names to be used as features.
        target_column (str): The name of the column to be used as the target variable.

    Returns:
        tuple: A tuple containing:
               - train_feature (np.array): Features for the training set.
               - train_target (np.array): Target for the training set.
               - test_feature (np.array): Features for the test set.
               - test_target (np.array): Target for the test set.
               - predefined_split_index (PredefinedSplit): An index for cross-validation
                 that designates the last 30 days of training data as the validation set.
    """
    logging.info("Splitting data into training, validation, testing, and forecast sets.")

    train_length = len(data) - 30
    logging.info(f"Training set size: {train_length}, Test set size: 30.")

    train_data = data.head(train_length)
    test_data = data.tail(30)
    
    train_feature = train_data[feature_columns].values
    train_target = train_data[target_column].values
    test_feature = test_data[feature_columns].values
    test_target = test_data[target_column].values
    logging.info("Succesfully splitted the data into training and testing sets")

    split_index = np.full(len(train_feature), -1, dtype=int)
    split_index[-30:] = 0
    predefined_split_index = PredefinedSplit(test_fold=split_index)
    logging.info("Sucessfully created PredefinedSplit for time-series cross-validation.")

    return train_feature, train_target, test_feature, test_target, predefined_split_index

def initialize_and_fit_model(train_feature: np.array, train_target: np.array, predefined_split_index: PredefinedSplit):
    """
    Initializes, tunes, and fits a CatBoost Classifier using Bayesian Optimization.

    This function uses BayesSearchCV to efficiently search for the optimal
    hyperparameters for a CatBoost model. It validates performance using a
    predefined time-series split and fits the best-found model on the entire
    training dataset.

    Args:
        train_feature (np.array): The feature set for training.
        train_target (np.array): The target variable for training.
        predefined_split_index (PredefinedSplit): The cross-validation strategy.

    Returns:
        CatBoostClassifier: The best-performing model found by the search.
    """
    logging.info("Initializing CatBoost model and starting hyperparameter tuning with BayesSearchCV.")

    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        logging_level='Silent'
    )

    search_spaces = {
        'depth': Integer(1, 5),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'iterations': Integer(150, 300),
        'l2_leaf_reg': Real(0.5, 3.0)
    }
    logging.info(f"Hyperparameter search space defined: {search_spaces}")
 
    scoring_method = 'roc_auc'
    if len(np.unique(train_target[-30:])) == 1:
        scoring_method = 'accuracy'
    logging.info(f"Using '{scoring_method}' as the scoring metric for hyperparameter tuning.")

    hyper_tune_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=25,
        cv=predefined_split_index,
        scoring=scoring_method,
        n_jobs=-1,
        verbose=0
    )

    logging.info("Fitting BayesSearchCV... This may take some time.")
    hyper_tune_search.fit(train_feature, train_target)
    logging.info("Hyperparameter tuning complete.")

    logging.info(f"Best parameters found: {hyper_tune_search.best_params_}")
    logging.info(f"Best cross-validated {scoring_method}: {hyper_tune_search.best_score_:.4f}")

    best_model = hyper_tune_search.best_estimator_

    return best_model

def calculate_classification_metrics(target_true: np.array, target_pred: np.array):
    """
    Calculates key classification metrics for a binary prediction task.

    Args:
        target_true (np.array): The ground truth labels.
        target_pred (np.array): The predicted labels from the model.

    Returns:
        tuple: A tuple containing accuracy, precision for both classes, and recall for both classes.
    """
    accuracy = accuracy_score(target_true, target_pred)
    precision_up_trend = precision_score(target_true, target_pred, pos_label='Up Trend', zero_division=0)
    precision_down_trend = precision_score(target_true, target_pred, pos_label='Down Trend', zero_division=0)
    recall_up_trend = recall_score(target_true, target_pred, pos_label='Up Trend', zero_division=0)
    recall_down_trend = recall_score(target_true, target_pred, pos_label='Down Trend', zero_division=0)

    return accuracy, precision_up_trend, precision_down_trend, recall_up_trend, recall_down_trend

def calculate_gini(target_true: np.array, target_pred_proba: np.array):
    """
    Calculates the Gini coefficient from the model's prediction probabilities.

    The Gini coefficient is a common metric for evaluating binary classification
    models and is derived from the Area Under the ROC Curve (AUC).
    Formula: Gini = 2 * AUC - 1

    Args:
        target_true (np.array): The true labels of the target variable.
        target_pred_proba (np.array): The predicted probabilities for each class.

    Returns:
        float: The calculated Gini coefficient, or 0.0 if AUC cannot be calculated.
    """
    try:
        positive_class_index = np.where(np.unique(target_true) == 'Up Trend')[0][0]
        auc = roc_auc_score(target_true, target_pred_proba[:, positive_class_index])
        gini = 2 * auc - 1
    except (ValueError, IndexError):
        logging.warning("Could not calculate Gini coefficient (likely only one class in target). Returning 0.0.")
        gini = 0.0

    return gini

def _measure_model_performance(model, feature: np.array, target: np.array, dataset_name: str):
    """
    (Internal Helper) Measures and reports the performance of the model on a given dataset.

    Args:
        model: The trained classifier model.
        feature (np.array): The feature set (e.g., train_feature or test_feature).
        target (np.array): The corresponding true target labels.
        dataset_name (str): The name of the dataset (e.g., "Training", "Testing").

    Returns:
        dict: A dictionary containing all calculated performance metrics.
    """
    logging.info(f"Evaluating model performance on the {dataset_name} set.")
    target_pred = model.predict(feature)
    target_pred_proba = model.predict_proba(feature)

    accuracy, prec_up, prec_down, rec_up, rec_down = calculate_classification_metrics(target, target_pred)
    gini = calculate_gini(target, target_pred_proba)

    all_metrics = {
        'Accuracy': [accuracy],
        'Precision Up Trend': [prec_up],
        'Precision Down Trend': [prec_down],
        'Recall Up Trend': [rec_up],
        'Recall Down Trend': [rec_down],
        'Gini': [gini]
    }
    logging.info(f"{dataset_name} Set Performance - Accuracy: {accuracy:.4f}, Gini: {gini:.4f}")

    return all_metrics

def measure_model_performance(model, train_feature: np.array, train_target: np.array, test_feature: np.array, test_target: np.array):
    """
    Evaluates and prints the model's performance on both training and testing data.

    This function provides a comprehensive view of the model's effectiveness by showing
    how it performs on the data it was trained on versus new, unseen data. This is
    essential for assessing potential overfitting.

    Args:
        model: The trained classifier model.
        train_feature, train_target: The training data.
        test_feature, test_target: The testing data.

    Returns:
        tuple: A tuple of dictionaries containing metrics for the training and testing sets.
    """
    train_metrics = _measure_model_performance(model, train_feature, train_target, "Training")
    test_metrics = _measure_model_performance(model, test_feature, test_target, "Testing")
    return train_metrics, test_metrics

def develop_model(prepared_data: pd.DataFrame, target_column: str):
    """
    Main orchestration function for the entire model development process.

    This function loads the feature names, splits the data, tunes the model,
    and evaluates its final performance.

    Args:
        prepared_data (pd.DataFrame): The fully prepared data from the previous step.
        target_column (str): The name of the target variable column.

    Returns:
        tuple: A tuple containing:
               - model (CatBoostClassifier): The final, trained model.
               - train_metrics (dict): Performance metrics on the training set.
               - test_metrics (dict): Performance metrics on the testing set.
    """
    logging.info(f"--- Starting Model Development for Target: '{target_column}' ---")

    feature_file = 'modelDevelopment/technical_indicator_features.txt'
    logging.info(f"Loading feature names from '{feature_file}'.")
    with open(feature_file, "r") as file:
        feature_columns = [line.strip() for line in file]

    train_feature, train_target, test_feature, test_target, cv_split = \
        split_data_to_train_val_test(prepared_data, feature_columns, target_column)

    model = initialize_and_fit_model(train_feature, train_target, cv_split)

    train_metrics, test_metrics = measure_model_performance(
        model, train_feature, train_target, test_feature, test_target
    )

    logging.info(f"--- Model Development for '{target_column}' Finished Successfully ---")
    return model, train_metrics, test_metrics