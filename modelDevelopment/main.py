import logging
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from catboost import CatBoostClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# --- Logging Configuration ---
# Configure the logger for clear, informative output during the model development process.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def split_data_to_train_val_test(data: pd.DataFrame, feature_columns: list, target_column: str):
    """
    Splits time-series data into training, testing, and forecasting sets.

    This function implements a time-based split crucial for financial forecasting:
    - Test Set: The most recent 30 days of data with valid targets.
    - Training Set: All data preceding the test set.
    - Validation Set (for Hyperparameter Tuning): The last 30 days of the training set.
    - Forecast Set: Data points at the end of the series where the target is unknown (NaN).

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
               - forecast_feature (np.array): Features for making future predictions.
    """
    logging.info("Splitting data into training, validation, testing, and forecast sets.")

    # Determine the number of rows at the end that have no target value.
    forecast_length = data[target_column].isna().sum()
    logging.info(f"Found {forecast_length} data points for future forecasting.")

    # Define the split point for the main training set.
    train_length = len(data) - 30 - forecast_length
    logging.info(f"Training set size: {train_length}, Test set size: 30.")

    # Slice the DataFrame into the respective sets.
    train_data = data.head(train_length)
    test_data = data.tail(30 + forecast_length).head(30)
    forecast_data = data.tail(forecast_length)

    # Extract feature and target numpy arrays for each set.
    train_feature = train_data[feature_columns].values
    train_target = train_data[target_column].values
    test_feature = test_data[feature_columns].values
    test_target = test_data[target_column].values
    forecast_feature = forecast_data[feature_columns].values

    # --- Create PredefinedSplit for Time-Series Cross-Validation ---
    # This is a crucial step for hyperparameter tuning on time-series data.
    # It ensures that we validate our model on the most recent portion of the
    # training data, simulating a real-world forecasting scenario.
    # We create an index where:
    # -1 indicates a sample is for training.
    #  0 indicates a sample is for validation/testing.
    split_index = np.full(len(train_feature), -1, dtype=int)
    split_index[-30:] = 0  # Mark the last 30 days of the training set as the validation fold.
    predefined_split_index = PredefinedSplit(test_fold=split_index)
    logging.info("Created PredefinedSplit for time-series cross-validation.")

    return train_feature, train_target, test_feature, test_target, predefined_split_index, forecast_feature

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

    # Initialize the CatBoost Classifier with baseline settings.
    model = CatBoostClassifier(
        loss_function='Logloss',  # Standard loss function for binary classification.
        eval_metric='AUC',        # Area Under Curve is a robust metric for evaluation.
        random_seed=42,           # For reproducibility.
        logging_level='Silent'    # Suppresses CatBoost's verbose output during search.
    )

    # Define the hyperparameter search space for Bayesian optimization.
    # We use both Integer and Real spaces for different parameter types.
    search_spaces = {
        'depth': Integer(1, 5),                                # Max depth of the trees.
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'), # Controls the step size.
        'iterations': Integer(150, 300),                       # Number of trees to build.
        'l2_leaf_reg': Real(0.5, 3.0)                          # L2 regularization strength.
    }
    logging.info(f"Hyperparameter search space defined: {search_spaces}")

    # Dynamically choose the scoring method. If the validation set has only one class,
    # ROC AUC is undefined, so we fall back to accuracy.
    scoring_method = 'roc_auc'
    if len(np.unique(train_target[-30:])) == 1:
        scoring_method = 'accuracy'
    logging.info(f"Using '{scoring_method}' as the scoring metric for hyperparameter tuning.")

    # Set up the Bayesian Search with our model, search space, and time-series CV.
    hyper_tune_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=100,                  # Number of parameter combinations to try.
        cv=predefined_split_index,  # Our custom time-series cross-validation split.
        scoring=scoring_method,
        n_jobs=-1,                  # Use all available CPU cores.
        random_state=42,
        verbose=0
    )

    # Run the search to find the best hyperparameters.
    logging.info("Fitting BayesSearchCV... This may take some time.")
    hyper_tune_search.fit(train_feature, train_target)
    logging.info("Hyperparameter tuning complete.")

    # Log the results of the search.
    logging.info(f"Best parameters found: {hyper_tune_search.best_params_}")
    logging.info(f"Best cross-validated {scoring_method}: {hyper_tune_search.best_score_:.4f}")

    # The best model is automatically refitted on the entire training data.
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
    # Calculate precision for each class separately.
    precision_up_trend = precision_score(target_true, target_pred, pos_label='Up Trend', zero_division=0)
    precision_down_trend = precision_score(target_true, target_pred, pos_label='Down Trend', zero_division=0)
    # Calculate recall for each class separately.
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
        # The positive class ('Up Trend') is typically the second column.
        # We need to find its index to correctly calculate AUC.
        positive_class_index = np.where(np.unique(target_true) == 'Up Trend')[0][0]
        auc = roc_auc_score(target_true, target_pred_proba[:, positive_class_index])
        gini = 2 * auc - 1
    except (ValueError, IndexError):
        # Handle cases where AUC cannot be computed (e.g., only one class present).
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
    # Get the model's class predictions and probability predictions.
    target_pred = model.predict(feature)
    target_pred_proba = model.predict_proba(feature)

    # Calculate standard classification metrics.
    accuracy, prec_up, prec_down, rec_up, rec_down = calculate_classification_metrics(target, target_pred)
    # Calculate the Gini coefficient.
    gini = calculate_gini(target, target_pred_proba)

    # Compile all metrics into a dictionary for easy reporting.
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
    # Calculate performance metrics for both sets using the helper function.
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

    # --- Step 1: Load Feature Names ---
    feature_file = 'modelDevelopment/technical_indicator_features.txt'
    try:
        logging.info(f"Loading feature names from '{feature_file}'.")
        with open(feature_file, "r") as file:
            feature_columns = [line.strip() for line in file]
        logging.info(f"Loaded {len(feature_columns)} features.")
    except FileNotFoundError:
        logging.error(f"Feature file not found at '{feature_file}'. Aborting.")
        return None, None, None

    # --- Step 2: Split Data ---
    train_feature, train_target, test_feature, test_target, cv_split, _ = \
        split_data_to_train_val_test(prepared_data, feature_columns, target_column)

    # --- Step 3: Initialize, Tune, and Fit Model ---
    model = initialize_and_fit_model(train_feature, train_target, cv_split)

    # --- Step 4: Evaluate Final Model ---
    train_metrics, test_metrics = measure_model_performance(
        model, train_feature, train_target, test_feature, test_target
    )

    logging.info(f"--- Model Development for '{target_column}' Finished Successfully ---")
    return model, train_metrics, test_metrics