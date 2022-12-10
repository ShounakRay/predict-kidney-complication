# @Author: shounak
# @Date:   2022-11-22T23:18:49-08:00
# @Email:  shounak@stanford.edu
# @Filename: models.py
# @Last modified by:   shounak
# @Last modified time: 2022-12-10T14:24:12-08:00

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from util import shuffle_data, nan_cols, _feed_through_model, split_data
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
import numpy as np
# import icd10
import json
# import requests
from bs4 import BeautifulSoup
# from collections import Counter

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# TODO: Implement StandardScaler for normalization
# TODO: Try out Sparse Regularization

# NOTES: https://stats.stackexchange.com/questions/497050/how-big-a-difference-for-test-train-rmse-is-considered-as-overfit

_ = """
################################################################################
############################# SOME HYPER-PARAMETERS ############################
################################################################################
"""
TRAIN_PROPORTION = 0.9

_ = """
################################################################################
################################ LOCAL FUNCTIONS ###############################
################################################################################
"""


def init_model():
    # DATA IS ALREADY EXTERNALLY SCALED
    model = LinearRegression()
    # with_mean=True since we're sparse represented as array
    # model = make_pipeline(StandardScaler(with_mean=True), LinearRegression())
    return model


def train_model(model, TRAIN_DF, plot=False):
    # Train Model
    design_matrix = TRAIN_DF[:, :-1].copy()
    prediction_column = TRAIN_DF[:, -1].copy()
    model.fit(design_matrix, prediction_column)
    # model.score(design_matrix, prediction_column) # Equiv to code below for getting r**2 value
    r, r_sq, std_err, rmse, _ = _feed_through_model(model, design_matrix, prediction_column,
                                                    plot, dataset_type='Training')

    return r, r_sq, std_err, rmse, _


def test_model(model, TEST_DF, plot=False):
    # Test Model
    test_design_matrix = TEST_DF[:, :-1]
    test_prediction_column = TEST_DF[:, -1]

    r, r_sq, std_err, rmse, _ = _feed_through_model(model, test_design_matrix, test_prediction_column,
                                                    plot, dataset_type='Testing')

    return r, r_sq, std_err, rmse, _


def load_data():
    with open('Data/Intermediate/training_tracker.json', 'r') as fp:
        training_tracker = {float(k): v for k, v in json.load(fp).items()}
    with open('Data/Intermediate/test_tracker.json', 'r') as fp:
        test_tracker = {float(k): v for k, v in json.load(fp).items()}
    with open('Data/Intermediate/training_std_err.json', 'r') as fp:
        training_std_err = {float(k): v for k, v in json.load(fp).items()}
    with open('Data/Intermediate/testing_std_err.json', 'r') as fp:
        testing_std_err = {float(k): v for k, v in json.load(fp).items()}
    with open('Data/Intermediate/training_rmse.json', 'r') as fp:
        training_rmse = {float(k): v for k, v in json.load(fp).items()}
    with open('Data/Intermediate/testing_rmse.json', 'r') as fp:
        testing_rmse = {float(k): v for k, v in json.load(fp).items()}

    return training_tracker, test_tracker, training_std_err, testing_std_err, training_rmse, testing_rmse


def save_data(training_tracker, test_tracker, training_std_err,
              testing_std_err, training_rmse, testing_rmse):
    with open('Data/Intermediate/training_tracker.json', 'w') as fp:
        json.dump(training_tracker, fp)
    with open('Data/Intermediate/test_tracker.json', 'w') as fp:
        json.dump(test_tracker, fp)
    with open('Data/Intermediate/training_std_err.json', 'w') as fp:
        json.dump(training_std_err, fp)
    with open('Data/Intermediate/testing_std_err.json', 'w') as fp:
        json.dump(testing_std_err, fp)
    with open('Data/Intermediate/training_rmse.json', 'w') as fp:
        json.dump(training_rmse, fp)
    with open('Data/Intermediate/testing_rmse.json', 'w') as fp:
        json.dump(testing_rmse, fp)


def run_many_splits(ALL_DATA, last_model=None, on='TRAIN-TEST-SPLIT', DEFAULT_PROP=0.7):
    training_tracker, test_tracker, training_std_err, testing_std_err = {}, {}, {}, {}
    training_rmse, testing_rmse = {}, {}

    def perform_run(ALL_DATA, last_model=None, prop=None, percent=None):
        VAR_TO_TRACK = prop
        if percent is not None:
            assert last_model is not None
            ALL_DATA = reduced_dataset(ALL_DATA, last_model, REMOVE_THIS_PERCENT=percent)
            VAR_TO_TRACK = percent
        # Split data
        TRAIN_DF, TEST_DF = split_data(ALL_DATA.copy(), prop)
        # Train model
        model = init_model()
        r_train, r_sq_train, std_err_train, rmse_train, _ = train_model(model, TRAIN_DF, plot=False)
        r_test, r_sq_test, std_err_test, rmse_test, _ = test_model(model, TEST_DF, plot=False)
        # Save information
        training_tracker[VAR_TO_TRACK], test_tracker[VAR_TO_TRACK] = r_train, r_test
        training_std_err[VAR_TO_TRACK], testing_std_err[VAR_TO_TRACK] = std_err_train, std_err_test
        training_rmse[VAR_TO_TRACK], testing_rmse[VAR_TO_TRACK] = rmse_train, rmse_test

    if on == 'TRAIN-TEST-SPLIT':
        for prop in np.array(range(500, 950 + 1)) / 1000:
            perform_run(ALL_DATA, last_model=None, prop=prop, percent=None)
        return training_tracker, test_tracker, training_std_err, testing_std_err, training_rmse, testing_rmse
    elif on == 'PERCENT_REMOVE':
        for percent in range(20, 100):
            perform_run(ALL_DATA, last_model=last_model, prop=DEFAULT_PROP, percent=percent)
        return training_tracker, test_tracker, training_std_err, testing_std_err, training_rmse, testing_rmse


def error_analysis(training_tracker1=None, test_tracker1=None, metric1='r-value',
                   training_metric2=None, testing_metric2=None, metric2='RMSE',
                   xlabel='Percentage Training Data', xline=None, save=True):
    """Visualize accuracy results."""
    scaler = 0.75
    fig, ax1 = plt.subplots(figsize=(16 * scaler, 10 * scaler))
    ax1.set_title(f'Linear Regression:\nImpact of {xlabel} on {metric1} and {metric2}')
    # First variable
    _ = ax1.set_xlabel(xlabel)
    _ = ax1.set_ylabel(metric1)
    _ = ax1.plot(training_tracker1.keys(), training_tracker1.values(),
                 label=f'Training {metric1}', color='darkorchid')
    _ = ax1.plot(test_tracker1.keys(), test_tracker1.values(),
                 label=f'Testing {metric1}', color='orangered')
    # Second variable
    ax2 = ax1.twinx()
    _ = ax2.set_ylabel(metric2)
    _ = ax2.plot(training_metric2.keys(), training_std_err.values(), ':',
                 label=f'Training {metric2}', color='darkorchid')
    _ = ax2.plot(testing_std_err.keys(), testing_metric2.values(), ':',
                 label=f'Testing {metric2}', color='orangered')
    _ = fig.legend(bbox_to_anchor=(0.5, -0.05), loc="lower center", ncol=4)
    # Line
    if xline is not None:
        _ = ax1.axvline(x=xline, linestyle='--')
    _ = plt.tight_layout()
    if save:
        _ = fig.savefig(f'Images/LinReg/LinReg_Error_Analysis_{metric1}_{metric2}_{xlabel}.png', bbox_inches='tight')
    _ = plt.show()


def really_check_single(ALL_DATA, iters=100, prop=0.7, plot=False):
    # Stick with train_proportion=0.7
    training_results, testing_results = [], []
    for sim in range(iters):
        ALL_DATA = shuffle_data(ALL_DATA)
        TRAIN_DF, TEST_DF = split_data(ALL_DATA.copy(), TRAIN_PROPORTION=prop)
        # Train model
        model = init_model()
        training_results.append(train_model(model, TRAIN_DF, plot=plot))
        testing_results.append(test_model(model, TEST_DF, plot=plot))
        # r_test, r_sq_test, std_err_test
    training_results = np.array(training_results)
    testing_results = np.array(testing_results)
    return training_results, testing_results


def load_check_single_70():
    with open("Data/Intermediate/training_results_70.json", 'r') as f:
        # f.write(str(training_results.tolist()))
        training_results = json.load(f)
    with open("Data/Intermediate/testing_results_70.json", 'r') as f:
        # f.write(str(testing_results.tolist()))
        testing_results = json.load(f)
    return np.array(training_results), np.array(testing_results)


def get_procedure_code(code):
    BeautifulSoup(data).find

# training_results, testing_results = load_check_single_70()
#
# def visualize_check_single(training_results, testing_results):
#     # _ = plt.hist(testing_results[:, 2], bins=20)
#     pass


def reduced_dataset(ALL_DATA, model, REMOVE_THIS_PERCENT=20, plot=False):
    # Calculate Feature Importance
    # print(f"Number of parameters: {model.coef_.shape[0]}")
    feature_importances = dict(zip(list(range(model.coef_.shape[0])), abs(model.coef_)))
    if plot:
        _ = plt.figure(figsize=(15, 10))
        _ = plt.bar(feature_importances.keys(), feature_importances.values())
        _ = plt.show()
        _ = plt.figure(figsize=(15, 10))
        _ = plt.hist(feature_importances.values(), bins=30)
        _ = plt.show()
    # Large values means great importance, small values means low importance
    # Anything below this threshold must be removed
    threshold = np.percentile(list(feature_importances.values()), REMOVE_THIS_PERCENT)
    keeping = {k: v for k, v in feature_importances.items() if v > threshold}
    BETTER_DATA = ALL_DATA[list(keeping.keys()) + ['PRED']].copy()
    # print(f"Removed {len(feature_importances) - len(keeping)} features from list")
    return BETTER_DATA


_ = """
################################################################################
################################ PRE-PROCESSING ################################
################################################################################
"""
# Load Data
ALL_DATA = pd.read_csv('Data/Merged Complete/Core_Dataset_SUPERVISED.csv').infer_objects().drop('Unnamed: 0', axis=1)
assert len(nan_cols(ALL_DATA)) == 0
# Shuffle
ALL_DATA = shuffle_data(ALL_DATA)
# Standardization
arr = ALL_DATA.values
scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler.fit(arr)
ALL_DATA = pd.DataFrame(scaler.transform(arr))
last_col = list(ALL_DATA)[-1]
ALL_DATA['PRED'] = ALL_DATA[last_col]
ALL_DATA.drop(last_col, axis=1, inplace=True)

PRED_RANGE = (ALL_DATA['PRED'].min(), ALL_DATA['PRED'].max())


_ = """
################################################################################
################################## SINGLE-RUN ##################################
################################################################################
"""

"""Train a "seed" model. 70/30 split, using all the features."""
TRAIN_DF, TEST_DF = split_data(ALL_DATA.copy(), 0.7)
# Train SEED_MODEL
SEED_MODEL = init_model()
r_train, r_sq_train, std_err_train, rmse_train, _ = train_model(SEED_MODEL, TRAIN_DF, plot=True)
r_test, r_sq_test, std_err_test, rmse_test, _ = test_model(SEED_MODEL, TEST_DF, plot=True)

"""
Using a 70-30 TRAIN-TEST split, what percent of features should we remove?
AKA, what percent:
– Minimizes RMSE and STD_ERR
– Maximizes R-VALUE
"""
(training_tracker, test_tracker, training_std_err,
 testing_std_err, training_rmse, testing_rmse) = run_many_splits(ALL_DATA,
                                                                 last_model=SEED_MODEL,
                                                                 on='PERCENT_REMOVE',
                                                                 DEFAULT_PROP=0.7)
error_analysis(training_tracker1=training_tracker, test_tracker1=test_tracker, metric1='r-value',
               training_metric2=testing_rmse, testing_metric2=testing_rmse, metric2='RMSE',
               xlabel='Percent Features Removed', xline=85, save=True)
error_analysis(training_tracker1=training_rmse, test_tracker1=testing_rmse, metric1='RMSE',
               training_metric2=training_std_err, testing_metric2=testing_std_err, metric2='Std. Error',
               xlabel='Percent Features Removed', xline=85, save=True)


"""
We noticed that removing 85% of features maximizes fit and minimizes error.
Removing 85% of features, what's the TRAIN-TEST split should we adopt?
AKA, what split:
– Minimizes RMSE and STD_ERR
– Maximizes R-VALUE
"""
BETTER_DATA = reduced_dataset(ALL_DATA, SEED_MODEL, REMOVE_THIS_PERCENT=85)
(training_tracker, test_tracker, training_std_err,
 testing_std_err, training_rmse, testing_rmse) = run_many_splits(BETTER_DATA,
                                                                 last_model=None,
                                                                 on='TRAIN-TEST-SPLIT',
                                                                 DEFAULT_PROP=None)
error_analysis(training_tracker1=training_tracker, test_tracker1=test_tracker, metric1='r-value',
               training_metric2=training_rmse, testing_metric2=testing_rmse, metric2='RMSE',
               xlabel='Percent Examples in Training Set', xline=0.6, save=True)
error_analysis(training_tracker1=training_rmse, test_tracker1=testing_rmse, metric1='RMSE',
               training_metric2=training_std_err, testing_metric2=testing_std_err, metric2='Std. Error',
               xlabel='Percent Examples in Training Set', xline=0.6, save=True)

BETTER_DATA.to_csv('Data/Better_Data.csv')

""" Now we're going to double check our results on a single run """
BETTER_TRAIN_DF, BETTER_TEST_DF = split_data(BETTER_DATA.copy(), 0.6)
# Train model
model = init_model()
r_train, r_sq_train, std_err_train, rmse_train, _ = train_model(model, BETTER_TRAIN_DF, plot=True)
r_test, r_sq_test, std_err_test, rmse_test, _ = test_model(model, BETTER_TEST_DF, plot=True)

""" FEATURE IMPORTANCE """
# feature_importances = dict(zip(list(range(model.coef_.shape[0])), abs(model.coef_)))
# linreg_ft_imps = dict(zip(mapping.values(), feature_importances.values()))
# _ = plt.hist(linreg_ft_imps.values(), bins=20)
# top_percentile = np.percentile(list(linreg_ft_imps.values()), 75)
# [ft for ft, imp in linreg_ft_imps.items() if imp >= top_percentile]

""" Now we're going to double check our results using CROSS VALIDATION """
training_results, testing_results = really_check_single(BETTER_DATA, iters=1000, prop=0.6, plot=False)
avg_r_train, avg_r_sq_train, avg_std_err_train, avg_rmse_train = (training_results[:, 0].mean(),
                                                                  training_results[:, 1].mean(),
                                                                  training_results[:, 2].mean(),
                                                                  training_results[:, 3].mean())
avg_r_test, avg_r_sq_test, avg_std_err_test, avg_rmse_test = (testing_results[:, 0].mean(),
                                                              testing_results[:, 1].mean(),
                                                              testing_results[:, 2].mean(),
                                                              testing_results[:, 3].mean())

"""Original Results"""
training_results, testing_results = really_check_single(ALL_DATA, iters=10, prop=0.7, plot=False)
avg_r_train, avg_r_sq_train, avg_std_err_train, avg_rmse_train = (training_results[:, 0].mean(),
                                                                  training_results[:, 1].mean(),
                                                                  training_results[:, 2].mean(),
                                                                  training_results[:, 3].mean())
avg_r_test, avg_r_sq_test, avg_std_err_test, avg_rmse_test = (testing_results[:, 0].mean(),
                                                              testing_results[:, 1].mean(),
                                                              testing_results[:, 2].mean(),
                                                              testing_results[:, 3].mean())

_ = """
################################################################################
############################ SPLIT SIZE VS. METRICS ############################
################################################################################
"""
# # # Conduct Simulation
# (training_tracker, test_tracker, training_std_err,
#  testing_std_err, training_rmse, testing_rmse) = run_many_splits(ALL_DATA)
# save_data(training_tracker, test_tracker, training_std_err,
#           testing_std_err, training_rmse, testing_rmse)
# # Load Already-Conducted Simulation Results
# training_tracker, test_tracker, training_std_err, testing_std_err, training_rmse, testing_rmse = load_data()
# # Plot errors and fit
# plt.plot(training_rmse.keys(), training_rmse.values())
# plt.plot(testing_rmse.keys(), testing_rmse.values())
# error_analysis(training_tracker, test_tracker, training_std_err,
#                testing_std_err, metric='Std Error', xlabel='Percent Removed')
# error_analysis(training_tracker, test_tracker, training_rmse, testing_rmse, metric2='RMSE', xlabel='Percent Removed')

_ = """
################################################################################
############################# ALT CROSS_VALIDATION #############################
################################################################################
"""

"""Observation:
A training_size of 70% seems to be ideal. After this point, the testing standard error
increases much faster, denoting lower model generalization to unseen data. The training r-value
also plateaus after this point.
"""

# # # Conduct [simulation] for single 0.7 split
# training_results, testing_results = really_check_single(ALL_DATA, iters=1, prop=0.7, plot=True)
#
# # # Load Already-Conducted Simulation Results for 0.7 split
# training_results, testing_results = load_check_single_70()
# # avg_r_train, avg_r_sq_train, avg_std_err_train = (training_results[:, 0].mean(),
# #                                                   training_results[:, 1].mean(),
# #                                                   training_results[:, 2].mean())
# # avg_r_test, avg_r_sq_test, avg_std_err_test = (testing_results[:, 0].mean(),
# #                                                testing_results[:, 1].mean(),
# #                                                testing_results[:, 2].mean())

_ = """
################################################################################
############################### CROSS VALIDATION ###############################
################################################################################
"""

# TRAIN_DF, TEST_DF = split_data(ALL_DATA.copy(), TRAIN_PROPORTION=0.7)
arr = BETTER_DATA.values
design_matrix = arr[:, :-1]
prediction_column = arr[:, -1]
# Train model
rmse = []
for k in range(2, len(BETTER_DATA)):
    model = init_model()
    cv = KFold(n_splits=k, random_state=1, shuffle=True)
    scores = cross_val_score(model, design_matrix, prediction_column, scoring='neg_mean_squared_error',
                             cv=cv, n_jobs=-1)
    rmse.append(np.mean(scores) * -1)

with open('Data/Intermediate/RMSE_Scores.json', 'w') as f:
    json.dump(rmse, f)

_ = plt.figure(figsize=(17, 10))
_ = plt.title('(Unconstrained) Relationship between number of folds and RMSE')
_ = plt.xlabel('Fold')
_ = plt.ylabel('RMSE')
_ = plt.plot(rmse)

_ = plt.figure(figsize=(17, 10))
_ = plt.title('(First-100) Relationship between number of folds and RMSE')
_ = plt.xlabel('Fold')
_ = plt.ylabel('RMSE')
_ = plt.plot(rmse[:100])
# EOF
