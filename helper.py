# EECS 445 - Fall 2024
# Project 1 - helper.py

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
#from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import project1
import matplotlib.pyplot as plt

def get_train_test_split() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
    """
    This function performs the following steps:
    - Reads in the data from data/labels.csv and data/files/*.csv (keep only the first 2,500 examples)
    - Generates a feature vector for each example
    - Aggregates feature vectors into a feature matrix (features are sorted alphabetically by name)
    - Performs imputation and normalization with respect to the population
    
    After all these steps, it splits the data into 80% train and 20% test. 
    
    The binary labels take two values:
        -1: survivor
        +1: died in hospital
    
    Returns the features and labels for train and test sets, followed by the names of features.
    """
    path =''
    df_labels = pd.read_csv(path+'data/labels.csv')
    df_labels = df_labels[:2000]
    IDs = df_labels['RecordID'][:2000]
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv(f'{path}data/files/{i}.csv')
    
    features = Parallel(n_jobs=16)(delayed(project1.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['In-hospital_death'].values

    X = project1.impute_missing_values(X)
    X = project1.normalize_feature_matrix(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    return X_train, y_train, X_test, y_test, feature_names

def get_challenge_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
    """
    This function is similar to get_train_test_split, except that:
    - It reads in all 10,000 training examples
    - It does not return labels for the 2,000 examples in the heldout test set
    You should replace your preprocessing functions (generate_feature_vector, 
    impute_missing_values, normalize_feature_matrix) with updated versions for the challenge 
    """
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv(f'data/files/{i}.csv')
    
    features = Parallel(n_jobs=16)(delayed(project1.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values
    X = project1.impute_missing_values(X)
    X = project1.normalize_feature_matrix(X)
    return X[:10000], y[:10000], X[10000:], feature_names
    


def generate_challenge_labels(y_label: npt.NDArray[np.float64], y_score: npt.NDArray[np.float64], uniqname: str) -> None:
    """
    Takes in `y_label` and `y_score`, which are two list-like objects that contain 
    both the binary predictions and raw scores from your linear classifier.
    Outputs the prediction to {uniqname}.csv. 
    
    Please make sure that you do not change the order of the test examples in the heldout set 
    since we will this file to evaluate your classifier.
    """
    pd.DataFrame({'label': y_label, 'risk_score': y_score}).to_csv(uniqname + '.csv', index=False)


#2.1c
def logistic_find_c_penalty(X, y, metric_list):
    """
    This function experiments combinations of c range and penaltieson logistic regression model
    The model was evaluated based on metrics in the metric list
    A table is printed.
    """
    results = []

    C_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    penalties = ['l1', 'l2']
    
    for metric in metric_list:
        result = project1.select_param_logreg(X, y, metric=metric, C_range = C_range, penalties=penalties)
        classifier = project1.get_classifier("logistic", penalty=result[1], C=result[0])
        mean_perform, min_perform, max_perform = project1.cv_performance(classifier, X, y, metric, 5)
        results.append([
            metric,
            result[0],
            result[1],
            f"{np.round(mean_perform, 5)} ({np.round(min_perform, 5)}, {np.round(max_perform, 5)})"
        ])

    df = pd.DataFrame(results, columns=["Performance Measure", "C", "Penalty", "Mean (Min, Max) CV Performance"])
    return df

#2.1d
def auroc_c_penalty( X_train, y_train, X_test, y_test, metric_list, C, penalty):
    results = []
    classifier = project1.get_classifier("logistic", C = C, penalty = penalty)
    classifier.fit(X_train, y_train)
    for metric in metric_list:
        median, lower, upper = project1.performance(classifier, X_test, y_test, metric, bootstrap = True)
        results.append([
            metric,
            median,
            f"({np.round(lower, 5)}, {np.round(upper, 5)})"
        ])
    df = pd.DataFrame(results, columns=["Performance Measure", "Median Performance", "95% Confidence Interval"])
    return df

#3.1b
def logistic_with_weights(X_train, y_train, X_test, y_test, metric_list, C, penalty):
    results = []
    clf = project1.get_classifier("logistic", penalty = penalty, C = C, class_weight={-1: 1, 1: 50})
    clf.fit(X_train, y_train)
    for metric in metric_list:
        median, lower, upper = project1.performance(clf, X_test, y_test, metric, bootstrap = True)
        results.append([
            metric,
            median,
            f"({np.round(lower, 5)}, {np.round(upper, 5)})"
        ])
    df = pd.DataFrame(results, columns=["Performance Measure", "Median", "95% Confidence Interval"])
    return df

#3.2a
def optimal_weights(X_train, y_train, X_test, y_test, metric_list, C, penalty):
    # we know from y_train, the approximated ratio between positive label and negative lable is 0.165
    # there I choose the weights Wn to be 16.5, Wp to be 100
    results = []
    clf = project1.get_classifier("logistic", penalty = penalty, C = C, class_weight={-1: 16.5, 1: 100})
    clf.fit(X_train, y_train)
    for metric in metric_list:
        mean, lower, upper = project1.cv_performance(clf, X_test, y_test, metric)
        results.append([
            metric,
            mean,
        ])
    df = pd.DataFrame(results, columns=["Performance Measure", "Mean"])
    return df

#3.3a
def draw_roc_curve(X_train, y_train, C, penalty):
    clf1 = project1.get_classifier("logistic", penalty = penalty, C = C, class_weight={-1: 1, 1: 1})
    clf2 = project1.get_classifier("logistic", penalty = penalty, C = C, class_weight={-1: 1, 1: 5})
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    y_scores1 = clf1.decision_function(X_train)
    y_scores2 = clf2.decision_function(X_train)

    fpr1, tpr1, _ = roc_curve(y_train, y_scores1)
    fpr2, tpr2, _ = roc_curve(y_train, y_scores2)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    plt.figure()
    plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'Wn=1, Wp=1 (AUC = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, color='red', lw=2, label=f'Wn=1, Wp=5 (AUC = {roc_auc2:.2f})')

    # Plotting diagonal line for reference
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2)

    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    plt.savefig("ROC_curve.png", dpi=200)
    plt.close()

#4b
def compare_logistic_ridge(X_train, y_train, X_test, y_test, metric_list):
    logistic = project1.get_classifier("logistic", C = 1.0, penalty = "l2")
    ridge = project1.get_classifier("squared_error", C = 1.0, kernel= "linear")
    logistic.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    results = []
    for metric in metric_list:
        median, lower, upper = project1.performance(logistic, X_test, y_test, metric, bootstrap = True)
        median2, lower2, upper2 = project1.performance(ridge, X_test, y_test, metric, bootstrap = True)
        results.append([
            metric,
            f"{np.round(median, 5)} ({np.round(lower, 5)}, {np.round(upper, 5)})",
            f"{np.round(median2, 5)} ({np.round(lower2, 5)}, {np.round(upper2, 5)})"
        ])
    df = pd.DataFrame(results, columns=["Performance Measure", "Log. Regression(median, 95%CI)", "Ridge Regression(median, 95%CI)"])
    return df

#4d
def eval_gamma(X, y, C, gammas, metric):
    results = []
    for gamma in gammas:
        classifier = project1.get_classifier("squared_error", C = C, gamma = gamma)
        mean_perform, min_perform, max_perform = project1.cv_performance(classifier, X, y, metric)
        results.append([
            gamma,
            f"{np.round(mean_perform, 5)} ({np.round(min_perform, 5)}, {np.round(max_perform, 5)})"
        ])
    df = pd.DataFrame(results, columns=["gamma", "Mean (Min, Max) CV Performance"])
    return df

#4e
def ridge_test_c_gamma(X_train, y_train, X_test, y_test, metric_list):
    results = []
    ridge = project1.get_classifier("squared_error", C = 0.1, gamma = 1.0)
    ridge.fit(X_train, y_train)
    for metric in metric_list:
        median, lower, upper = project1.performance(ridge, X_test, y_test, metric, bootstrap = True)
        results.append([
            metric,
            np.round(median, 5),
            f"({np.round(lower, 5)}, {np.round(upper, 5)})"
        ])
    df = pd.DataFrame(results, columns=["Performance Measure", "Median Performance", "(95% Confidence Interval)"])
    return df


