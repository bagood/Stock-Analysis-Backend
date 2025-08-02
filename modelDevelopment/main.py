import numpy as np
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score

def split_data_to_train_and_test(data, feature_columns, target_column):
    """
    Splits the dataset into training and testing sets based on a fixed time-series split.

    This function treats the last 30 data points as the test set and all preceding
    data points as the training set. This is a common approach for time-series
    data to ensure the model is tested on the most recent information.

    Args:
        data (pd.DataFrame): The complete DataFrame containing features and target.
        feature_columns (list): A list of column names to be used as features.
        target_column (str): The name of the column to be used as the target variable.

    Returns:
        tuple: A tuple containing the split data:
               (train_feature, train_target, test_feature, test_target).
    """
    # Define the split point: all data except the last 30 rows is for training.
    train_length = len(data) - 30
    
    # Create the training and testing DataFrames.
    train_data = data.head(train_length)
    test_data = data.tail(30)

    # Extract feature and target arrays for the training set.
    train_feature = train_data[feature_columns].values
    train_target = train_data[target_column].values.reshape(-1, 1)

    # Extract feature and target arrays for the testing set.
    test_feature = test_data[feature_columns].values
    test_target = test_data[target_column].values.reshape(-1, 1)

    return train_feature, train_target, test_feature, test_target

def initialize_and_fit_model(train_feature, train_target):
    """
    Initializes, tunes, and fits a CatBoost Classifier model using Randomized Search.

    This function sets up a CatBoost model and defines a hyperparameter search space.
    It then uses RandomizedSearchCV to efficiently find the best combination of
    hyperparameters based on cross-validated accuracy, and fits the best model
    on the entire training dataset.

    Args:
        train_feature (np.array): The feature set for training.
        train_target (np.array): The target variable for training.

    Returns:
        CatBoostClassifier: The best-performing model found by the search.
    """
    # Initialize the CatBoost Classifier with baseline settings.
    model = CatBoostClassifier(
        loss_function='Logloss',      # Loss function suitable for binary classification.
        eval_metric='AUC',       # Metric to evaluate during training.
        random_seed=42,               # Ensures reproducibility.
        logging_level='Silent'        # Suppresses verbose output during training.
    )
    
    # Define the distribution of hyperparameters to sample from during the search.
    param_dist = {
        'depth': randint(1, 5),                      # Tree depth.
        'learning_rate': uniform(0.01, 0.1),         # Step size shrinkage.
        'iterations': randint(150, 300),             # Number of boosting iterations (trees).
        'l2_leaf_reg': uniform(0.5, 3)               # L2 regularization strength.
    }

    # Set up the Randomized Search with cross-validation.
    random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=1000,                  # Number of parameter settings that are sampled.
            cv=3,                         # Number of cross-validation folds.
            scoring='roc_auc',            # Metric to optimize.
            n_jobs=-1,                    # Use all available CPU cores.
            random_state=42,              # Ensures reproducibility of the search.
            verbose=1                     # Prints progress updates.
        )

    # Run the randomized search to find the best hyperparameters.
    random_search.fit(train_feature, train_target)
    
    # Print the results of the hyperparameter search.
    print("Best parameters found by RandomizedSearchCV:")
    print(random_search.best_params_)
    print(f"\nBest cross-validated accuracy: {random_search.best_score_:.4f}")
    print("-" * 30 + "\n")
    
    # Get the best model found during the search.
    best_model = random_search.best_estimator_

    return best_model

def calculate_gini(target_true, target_pred_proba):
    """
    Calculates the Gini coefficient from the model's prediction probabilities.

    The Gini coefficient is a common metric for evaluating binary classification
    models and is derived from the Area Under the ROC Curve (AUC).
    
    Formula: Gini = 2 * AUC - 1

    Args:
        target_true (np.array): The true labels of the target variable.
        target_pred_proba (np.array): The predicted probabilities for each class,
                                      as returned by `model.predict_proba()`.

    Returns:
        float: The calculated Gini coefficient.
    """
    # Calculate the AUC score for the positive class (class 1).
    auc = roc_auc_score(target_true, target_pred_proba[:, 1])
    # Convert the AUC score to the Gini coefficient.
    gini = 2 * auc - 1
    
    return gini

def _measure_model_performance(model, feature, target):
    """
    (Internal Helper) Measures and reports the performance of the model on a given dataset.

    Args:
        model: The trained classifier model.
        feature (np.array): The feature set (e.g., train_feature or test_feature).
        target (np.array): The corresponding true target labels.

    Returns:
        tuple: A tuple containing the classification report (str) and the Gini score (float).
    """
    # Get the model's class predictions and probability predictions.
    target_pred = model.predict(feature)
    target_pred_proba = model.predict_proba(feature)
    
    # Generate a detailed classification report (precision, recall, f1-score).
    report = classification_report(target, target_pred)
    # Calculate the Gini coefficient.
    gini = calculate_gini(target, target_pred_proba)

    return report, gini
    
def measure_model_performance(model, train_feature, train_target, test_feature, test_target):
    """
    Evaluates and prints the model's performance on both training and testing data.

    This function provides a comprehensive view of the model's effectiveness by showing
    how it performs on the data it was trained on versus new, unseen data. This helps
    in assessing overfitting.

    Args:
        model: The trained classifier model.
        train_feature (np.array): The feature set for training.
        train_target (np.array): The target variable for training.
        test_feature (np.array): The feature set for testing.
        test_target (np.array): The target variable for testing.
    """
    # Calculate performance metrics for the training set.
    train_report, train_gini = _measure_model_performance(model, train_feature, train_target)
    # Calculate performance metrics for the testing set.
    test_report, test_gini = _measure_model_performance(model, test_feature, test_target)

    # Print the formatted performance reports.
    print('Model performance on training data')
    print('')
    print(train_report)
    print('    Gini: ', train_gini)
    print('')
    
    print('Model performance on testing data')
    print('')
    print(test_report)
    print('    Gini:  ', test_gini)
    print('')
    
    return
