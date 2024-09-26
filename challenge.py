# EECS 445 - Fall 2024
# Project 1 - challenge.py

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
#from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from helper import *
from project1 import *
from challenge_helper import *
from imblearn.combine import SMOTETomek
from sklearn.metrics import confusion_matrix

def get_train_test_split_challenge() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
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
    
    features = Parallel(n_jobs=16)(delayed(generate_feature_vector_challenge)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values

    X = impute_missing_values_challenge(X)
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
    
    features = Parallel(n_jobs=16)(delayed(generate_feature_vector_challenge)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values
    X = impute_missing_values_challenge(X)
    X = project1.normalize_feature_matrix(X)
    return X[:10000], y[:10000], X[10000:], feature_names

def main() -> None:
    print(f"Using Seed={seed}")
    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # step 1: load data
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split_challenge()
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    # step 2: select classifier
    # we choose logistic regression here because it handles binary classification well

    # run this to determine hyperparameters
    '''
    df = logistic_find_c_penalty_challenge(X_train, y_train, metric_list)
    print(df)
    '''
    
    # we discovered that the ratio of positive label to negative label in y_challenge
    '''
    count_ones = np.sum(y_challenge == 1)
    count_neg_ones = np.sum(y_challenge == -1)
    print(count_ones/count_neg_ones)
    '''
    
    # is 0.18203309692671396. We choose weights based on it
    penalty = "l2"
    C = 0.01
    weights = {-1: 18.20331, 1: 100}

    clf = get_classifier("logistic", penalty = penalty, C = C, class_weight = weights)

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_challenge, y_challenge)

    #step 3: fit the model
    clf.fit(X_challenge, y_challenge)

    #evaluate
    '''
    results = []
    for metric in metric_list:
        median, lower, upper = performance_challenge(clf, X_train, y_train, metric, bootstrap = True)
        results.append([
            metric,
            median,
            f"({np.round(lower, 5)}, {np.round(upper, 5)})"
        ])
    df = pd.DataFrame(results, columns=["Performance Measure", "Median", "95% Confidence Interval"])
    print(df)
    '''
    
    y_pred = clf.predict(X_challenge)  # Get the predictions on the training data (X_challenge)
    cm = confusion_matrix(y_challenge, y_pred, labels=[1, -1])  # Generate the confusion matrix
    
    print("\nConfusion Matrix (True labels: rows, Predicted labels: columns):")
    print(pd.DataFrame(cm, index=["True Positive (1)", "True Negative (-1)"], columns=["Predicted Positive (1)", "Predicted Negative (-1)"]))

    #step4: output testing result
    y_pred_prob = clf.predict_proba(X_heldout)[:, 1]  # Probabilities for class 1
    threshold = 0.6
    y_label = np.where(y_pred_prob >= threshold, 1, -1)
    y_score = clf.decision_function(X_heldout)

    generate_challenge_labels(y_label = y_label, y_score = y_score, uniqname="yueqiren")

    return

if __name__ == "__main__":
    main()





