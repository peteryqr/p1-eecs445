# EECS 445 - Fall 2024
# Project 1 - project1.py

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
import sklearn.metrics

config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = config["seed"]
np.random.seed(seed)


# Q1a
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: dataframe with columns [Time, Variable, Value]

    Returns:
        a dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'max_HR': 84, ...}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]

    # replace unknown values with NA
    df = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    for variable in static_variables:
        feature_dict[variable] = static.loc[static["Variable"] == variable, 'Value'].iloc[0]

    for variable in timeseries_variables:
        feature_dict["max_" + variable] = timeseries.loc[timeseries["Variable"] == variable, 'Value'].max() # take max

    return feature_dict


# Q1b
def impute_missing_values(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Args:
        X: (N, d) matrix. X could contain missing values
    
    Returns:
        X: (N, d) matrix. X does not contain any missing values
    """
    column_means = np.nanmean(X, axis = 0)
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            if np.isnan(X[j, i]):
                X[j, i] = column_means[i]
    assert not np.isnan(X).any(), "There are still NaN values in X after imputation."
    return X


# Q1c
def normalize_feature_matrix(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (N, d) matrix.
    
    Returns:
        X: (N, d) matrix. Values are normalized per column.
    """
    # for each column
    for i in range(X.shape[1]):
        col_min = np.min(X[:, i])
        col_max = np.max(X[:, i])
        # we assume min and max are not the same for any variable
        if col_max == col_min:
            X[:, i] = 0
        else:
            X[:, i] = (X[:, i] - col_min)/(col_max - col_min)
    return X


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function
    and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization (default: None).
        C: Regularization strength parameter (default: 1.0).
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression. 
            Default is 'rbf'.
        gamma (float): Kernel coefficient (default: 0.1).
    Returns:
        A classifier based on the specified arguments.
    """
    if loss == "logistic":
        # Logistic Regression without specifying solver or max_iter
        return LogisticRegression(
            penalty=penalty,            
            C=C,                        
            class_weight=class_weight,
            solver="liblinear", 
            fit_intercept=False, 
            random_state=seed,
            max_iter = 1000
        )
    elif loss == "squared_error":
        # Kernel Ridge Regression
        return KernelRidge(
            alpha=1.0 / (2 * C),   
            kernel=kernel,
            gamma=gamma      
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

#helper functions
def calculate_metric_logistic(X, y_true, clf_trained, metric):
    if metric == "accuracy":
        y_pred = clf_trained.predict(X)
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        y_pred = clf_trained.predict(X)
        return metrics.precision_score(y_true, y_pred, zero_division = 0)
    elif metric == "f1_score":
        y_pred = clf_trained.predict(X)
        return metrics.f1_score(y_true, y_pred, zero_division = 0)
    elif metric == "auroc":
        y_pred = clf_trained.decision_function(X)
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "average_precision":
        y_pred = clf_trained.decision_function(X)
        return metrics.average_precision_score(y_true, y_pred)
    elif metric == "sensitivity":
        y_pred = clf_trained.predict(X)
        return metrics.recall_score(y_true, y_pred)
    elif metric == "specificity":
        y_pred = clf_trained.predict(X)
        tn, fp, _, _ = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        raise ValueError("Unsupported metric")

def calculate_metric_ridge(X, y_true, clf_trained, metric):
    y_pred = clf_trained.predict(X)
    if metric == "accuracy":
        y_pred = np.where(y_pred >= 0, 1, -1)
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        y_pred = np.where(y_pred >= 0, 1, -1)
        return metrics.precision_score(y_true, y_pred, zero_division = 0)
    elif metric == "f1_score":
        y_pred = np.where(y_pred >= 0, 1, -1)
        return metrics.f1_score(y_true, y_pred, zero_division = 0)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "average_precision":
        return metrics.average_precision_score(y_true, y_pred)
    elif metric == "sensitivity":
        y_pred = np.where(y_pred >= 0, 1, -1)
        return metrics.recall_score(y_true, y_pred)
    elif metric == "specificity":
        y_pred = np.where(y_pred >= 0, 1, -1)
        tn, fp, _, _ = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        raise ValueError("Unsupported metric")


def performance(
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
        # Single sample performance
        if isinstance(clf_trained, KernelRidge):
            return calculate_metric_ridge(X, y_true, clf_trained, metric)
        else:
            return calculate_metric_logistic(X, y_true, clf_trained, metric)
    
    # bootstrap
    n_iterations = 1000
    bootstrap_metrics = []

    for i in range(n_iterations):
        # Resample the dataset (with replacement)
        X_resampled, y_resampled = resample(X, y_true)
        # train and alculate the metric on resampled dataset
        if isinstance(clf_trained, KernelRidge):
            score = calculate_metric_ridge(X_resampled, y_resampled, clf_trained, metric)
        else:
            score = calculate_metric_logistic(X_resampled, y_resampled, clf_trained, metric)
        bootstrap_metrics.append(score)

    # Calculate median and confidence intervals
    bootstrap_metrics = np.array(bootstrap_metrics)
    median = np.median(bootstrap_metrics)
    lower_bound = np.percentile(bootstrap_metrics, 2.5)
    upper_bound = np.percentile(bootstrap_metrics, 97.5)

    return median, lower_bound, upper_bound

# Q2.1a
def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    
    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
    
    Returns:
        a tuple containing (mean, min, max) 'cross-validation' performance across the k folds
    """
    # Set up StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    
    # List to store performance of each fold
    performance_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the classifier on the training data
        clf.fit(X_train, y_train)
        
        # Calculate the performance metric
        score = performance(clf, X_test, y_test, metric, bootstrap = False)
        performance_scores.append(score)
    
    # Calculate the mean, min, and max performance across folds
    mean_performance = np.mean(performance_scores)
    min_performance = np.min(performance_scores)
    max_performance = np.max(performance_scores)
    
    # Return a tuple (mean, min, max)
    return (mean_performance, min_performance, max_performance)


# Q2.1b
def select_param_logreg(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over
    
    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    optimal_performance = -np.inf  # To handle maximization of the metric
    optimal_c, optimal_penalty = 0, None
    for c in C_range:
        for penalty in penalties:
            classifier = get_classifier("logistic", penalty, c)
            
            # cv_performance function to evaluate performance (k-fold)
            mean_perform, min_perform, max_perform = cv_performance(classifier, X, y, metric, k)

            # Update optimal hyperparameters if performance improves
            if mean_perform > optimal_performance:
                optimal_c = c
                optimal_penalty = penalty
                optimal_performance = mean_perform

    return (optimal_c, optimal_penalty)

# Q4c
def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over
    
    Returns:
        The parameter value for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    print(f"RBF Kernel Ridge Regression Model Hyperparameter Selection based on {metric}:")
    optimal_performance = -np.inf  # To handle maximization of the metric
    optimal_c, optimal_gamma = 0, 0
    for c in C_range:
        for gamma in gamma_range:
            classifier = get_classifier("squared_error", C = c, gamma = gamma)
            
            # cv_performance function to evaluate performance (k-fold)
            mean_perform, min_perform, max_perform = cv_performance(classifier, X, y, metric, k)

            # Update optimal hyperparameters if performance improves
            if mean_perform > optimal_performance:
                optimal_c = c
                optimal_gamma = gamma
                optimal_performance = mean_perform

    return (optimal_c, optimal_gamma)

# Q2.1e
def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
    
    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        # elements of norm0 should be the number of non-zero coefficients for a given setting of C
        norm0 = []
        for C in C_range:
            clf = get_classifier("logistic", penalty = penalty, C = C)
            clf.fit(X, y)

            w = clf.coef_

            # TODO: Count number of nonzero coefficients/weights for setting of C
            #      and append count to norm0
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    # NOTE: plot will be saved in the current directory
    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()


def main() -> None:
    print(f"Using Seed={seed}")
    # Read data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_vector, impute_missing_values AND normalize_feature_matrix
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split()

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!

    #1.d
    '''
    mean_values = np.mean(X_train, axis=0)
    q75, q25 = np.percentile(X_train, [75 ,25], axis=0)
    iqr_values = q75 - q25
    df = pd.DataFrame({
    'mean': mean_values,
    'IQR': iqr_values
    }, index=[f'{feature}' for feature in feature_names])
    print(df)
    '''
    

    #2.1c
    '''
    optimal_hyperparameter = logistic_find_c_penalty(X_train, y_train, metric_list)
    print(optimal_hyperparameter)
    '''
    
    
    #2.1d
    '''
    table = auroc_c_penalty( X_train, y_train, X_test, y_test, metric_list, 1, "l1")
    print(table)
    '''
    
    
    #2.1e
    '''
    C_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    penalties = ['l1', 'l2']
    plot_weight(X_train, y_train, C_range= C_range, penalties = penalties)
    '''
    
    #2.1f
    '''
    clf = get_classifier("logistic", penalty = "l1", C = 1.0)
    clf.fit(X_train, y_train)
    coefficients = clf.coef_

    sorted_indices = np.argsort(coefficients)
    most_negative_indices = sorted_indices[0, :4]
    most_positive_indices = sorted_indices[0, -4:]

    print("Negative Coefficients: ")
    for i in most_negative_indices:
        print(f"{feature_names[i]}: {coefficients[0, i]}")

    print("Positive Coefficients: ")
    for i in most_positive_indices:
        print(f"{feature_names[i]}: {coefficients[0, i]}")
    '''

    #3.1b
    '''
    weighted_logistic = logistic_with_weights(X_train, y_train, X_test, y_test, metric_list, 1, "l2")
    print(weighted_logistic)
    '''
    
    #3.2a & b
    '''
    count_ones = np.sum(y_train == 1)
    count_neg_ones = np.sum(y_train == -1)
    print(count_ones/count_neg_ones)
    optimal_weight = optimal_weights(X_train, y_train, X_test, y_test, metric_list, 1, "l2")
    print(optimal_weight)
    '''
    
    
    
    #3.3a
    #draw_roc_curve(X_train, y_train, 1.0, "l2")

    #4.b
    '''
    df = compare_logistic_ridge(X_train, y_train, X_test, y_test, metric_list)
    print(df)
    '''

    #4.d
    '''
    C = 1.0
    gammas = [0.001, 0.01, 0.1, 1, 10, 100]
    metric = "auroc"
    df = eval_gamma(X_train, y_train, C=C, gammas = gammas, metric = metric)
    print(df)
    '''
    
    #4.e
    # first we find c and gamma that opimize auroc
    '''
    C = [0.01, 0.1, 1.0, 10, 100]
    gammas = [0.01, 0.1, 1, 10]
    print(select_param_RBF(X_train, y_train, metric = "auroc", C_range=C, gamma_range = gammas))
    '''
    
    # the resulting c is 0.1, and gamma is 1
    df = ridge_test_c_gamma(X_train, y_train, X_test, y_test, metric_list)
    print(df)


    return

    # Read challenge data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    # X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()


if __name__ == "__main__":
    main()
