# EECS 445 - Fall 2024
# Project 1 - challenge_helper.py

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = config["seed"]
np.random.seed(seed)

def generate_feature_vector_challenge(df: pd.DataFrame) -> dict[str, float]:
    """
    preprocess data
    Convert ICUType to four binary data
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]

    # replace unknown values with NA
    df = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    for variable in static_variables:
        if (variable == "ICUType"):
            icu_type_value = static.loc[static["Variable"] == variable, 'Value'].iloc[0]

            # Create four binary attributes for each ICUType (1, 2, 3, 4)
            feature_dict['CORONARY'] = 1 if icu_type_value == 1 else 0
            feature_dict['CARDIAC'] = 1 if icu_type_value == 2 else 0
            feature_dict['MEDICAL'] = 1 if icu_type_value == 3 else 0
            feature_dict['SURGICAL'] = 1 if icu_type_value == 4 else 0
        else:
            feature_dict[variable] = static.loc[static["Variable"] == variable, 'Value'].iloc[0]

    # Extract the first two digits (hours) and convert to a numeric value
    timeseries.loc[:, 'Time'] = timeseries['Time'].str[:2].astype(float)

    # Existing logic for filtering the first and second 24 hours
    first_24h = timeseries[timeseries['Time'] <= 24]
    later_24h = timeseries[(timeseries['Time'] > 24) & (timeseries['Time'] <= 48)]
    for variable in timeseries_variables:
        # Max, Min, Mean, and Std over the first 24 hours
        #feature_dict["max_24h_" + variable] = first_24h.loc[first_24h["Variable"] == variable, 'Value'].max()
        feature_dict["mean_24h_" + variable] = first_24h.loc[first_24h["Variable"] == variable, 'Value'].mean()
        #feature_dict["std_24h_" + variable] = first_24h.loc[first_24h["Variable"] == variable, 'Value'].std()

        # Max, Min, Mean, and Std over the later 24 hours
        feature_dict["max_later_24h_" + variable] = later_24h.loc[later_24h["Variable"] == variable, 'Value'].max()
        feature_dict["mean_later_24h_" + variable] = later_24h.loc[later_24h["Variable"] == variable, 'Value'].mean()
        #feature_dict["std_later_24h_" + variable] = later_24h.loc[later_24h["Variable"] == variable, 'Value'].std()
    return feature_dict

def select_param_logreg_challenge(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    
    param_grid = {
        'C': C_range,
        'penalty': penalties
    }

    optimal_performance = -np.inf  # To handle maximization of the metric
    optimal_c, optimal_penalty = 0, None
    weights = {-1: 18.20331, 1: 100}

    for c in C_range:
        for penalty in penalties:
            classifier = project1.get_classifier("logistic", penalty=penalty, C=c, class_weight = weights)
        
            # cv_performance function to evaluate performance (k-fold)
            mean_perform, min_perform, max_perform = project1.cv_performance(classifier, X, y, metric, k)

            # Update optimal hyperparameters if performance improves
            if mean_perform > optimal_performance:
                optimal_c = c
                optimal_penalty = penalty
                optimal_performance = mean_perform
    
    print(optimal_c, optimal_penalty)
    return (optimal_c, optimal_penalty)

def logistic_find_c_penalty_challenge(X, y, metric_list):
    """
    used to find optimal hyperparameters
    """
    results = []

    C_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    penalties = ['l1', 'l2']

    for metric in metric_list:
        result = select_param_logreg_challenge(X, y, metric=metric, C_range = C_range, penalties=penalties)
        results.append([
            metric,
            result[0],
            result[1],
        ])

    df = pd.DataFrame(results, columns=["Performance Measure", "C", "Penalty"])
    return df

def logistic_f1_find_c_penalty_challenge(X, y):
    """
    used to find optimal hyperparameters to imporve f1 score
    """
    results = []

    param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Range of regularization strengths
    'penalty': ['l1', 'l2']  # Different penalties
    }

    weights = {-1: 28.20331, 1: 100}

    # Initialize the Logistic Regression model with 'saga' solver to handle l1 and l2 penalties
    clf = LogisticRegression(class_weight=weights, max_iter=1000)

    # Initialize GridSearchCV with the logistic regression model
    grid_search = GridSearchCV(clf, param_grid, scoring='f1', cv=5, n_jobs=-1)

    # Fit the model to the data (assuming X_train and y_train are already defined)
    grid_search.fit(X, y)

    # Print the best parameters found by GridSearchCV
    print(f"Best parameters: {grid_search.best_params_}")


#helper functions
def calculate_metric_logistic_challenge(y_pred, y_true, y_score, metric):
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred, zero_division = 0)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_pred, zero_division = 0)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_score)
    elif metric == "average_precision":
        return metrics.average_precision_score(y_true, y_score)
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred)
    elif metric == "specificity":
        tn, fp, _, _ = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        raise ValueError("Unsupported metric")

def performance_challenge(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int64],
    metric: str = "accuracy",
    bootstrap: bool=True
) -> tuple[np.float64, np.float64, np.float64] | np.float64:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X, using 1,000 
    bootstrapped samples of the test set if bootstrap is set to True. Otherwise,
    returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.
    
    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision', 
                'sensitivity', and 'specificity')
    Returns:
        if bootstrap is True: the median performance and the empirical 95% confidence interval in np.float64
        if bootstrap is False: peformance 
    """
    if not bootstrap:
        y_pred_prob = clf_trained.predict_proba(X)[:, 1]  # Probabilities for class 1
        threshold = 0.6
        y_pred_custom_threshold = np.where(y_pred_prob >= threshold, 1, -1)
        y_score = clf_trained.decision_function(X)
        return calculate_metric_logistic_challenge(y_pred_custom_threshold, y_true, y_score, metric)
    
    # bootstrap
    n_iterations = 1000
    bootstrap_metrics = []

    for i in range(n_iterations):
        # Resample the dataset (with replacement)
        X_resampled, y_resampled = resample(X, y_true)
        # train and alculate the metric on resampled dataset
        y_pred_prob = clf_trained.predict_proba(X_resampled)[:, 1]  # Probabilities for class 1
        threshold = 0.6
        y_pred_custom_threshold = np.where(y_pred_prob >= threshold, 1, -1)
        y_score = clf_trained.decision_function(X_resampled)
        score = calculate_metric_logistic_challenge(y_pred_custom_threshold, y_resampled, y_score, metric)
        bootstrap_metrics.append(score)

    # Calculate median and confidence intervals
    bootstrap_metrics = np.array(bootstrap_metrics)
    median = np.median(bootstrap_metrics)
    lower_bound = np.percentile(bootstrap_metrics, 2.5)
    upper_bound = np.percentile(bootstrap_metrics, 97.5)

    return median, lower_bound, upper_bound

def impute_missing_values_challenge(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Args:
        X: (N, d) matrix. X could contain missing values
    
    Returns:
        X: (N, d) matrix. X does not contain any missing values
    """
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X)
    
    # Ensure there are no NaN values after imputation
    assert not np.isnan(X_train_imputed).any(), "There are still NaN values in X after imputation."
    
    return X_train_imputed