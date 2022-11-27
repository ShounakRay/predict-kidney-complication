# @Author: shounak
# @Date:   2022-11-22T23:18:49-08:00
# @Email:  shounak@stanford.edu
# @Filename: models.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-26T16:19:16-08:00

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from util import shuffle_data, nan_cols, _feed_through_model, split_data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import json

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# TODO: Implement StandardScaler for normalization
# TODO: Try out Sparse Regularization

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
    design_matrix = TRAIN_DF[:, :-1]
    prediction_column = TRAIN_DF[:, -1]
    model.fit(design_matrix, prediction_column)
    # model.score(design_matrix, prediction_column) # Equiv to code below for getting r**2 value
    r, r_sq, std_err, rmse, ALL_DATA = _feed_through_model(model, design_matrix, prediction_column,
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


def run_many_splits(ALL_DATA):
    training_tracker, test_tracker, training_std_err, testing_std_err = {}, {}, {}, {}
    training_rmse, testing_rmse = {}, {}
    for prop in np.array(range(500, 950 + 1)) / 1000:
        # Split data
        TRAIN_DF, TEST_DF = split_data(ALL_DATA, prop)
        # Train model
        model = init_model()
        r_train, r_sq_train, std_err_train, rmse_train, _ = train_model(model, TRAIN_DF, plot=False)
        r_test, r_sq_test, std_err_test, rmse_test, _ = test_model(model, TEST_DF, plot=False)
        # Save information
        training_tracker[prop], test_tracker[prop] = r_train, r_test
        training_std_err[prop], testing_std_err[prop] = std_err_train, std_err_test
        training_rmse[prop], testing_rmse[prop] = rmse_train, rmse_test

    return training_tracker, test_tracker, training_std_err, testing_std_err, training_rmse, testing_rmse


def error_analysis(training_tracker, test_tracker, training_metric, testing_metric, metric='RMSE'):
    # Visualize accuracy results
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax1.set_title(f'Linear Regression:\nImpact of training set size on r-value and {metric}')
    # First variable
    _ = ax1.set_xlabel("Percentage training data")
    _ = ax1.set_ylabel("r-value")
    _ = ax1.plot(training_tracker.keys(), training_tracker.values(),
                 label='Training r-value', color='darkorchid')
    _ = ax1.plot(test_tracker.keys(), test_tracker.values(),
                 label='Testing r-value', color='orangered')
    # Second variable
    ax2 = ax1.twinx()
    _ = ax2.set_ylabel(metric)
    _ = ax2.plot(training_metric.keys(), training_std_err.values(), ':',
                 label=f'Training {metric}', color='darkorchid')
    _ = ax2.plot(testing_std_err.keys(), testing_metric.values(), ':',
                 label=f'Testing {metric}', color='orangered')
    _ = fig.legend(bbox_to_anchor=(0.5, 0), loc="lower center", ncol=4)
    # Line
    _ = ax1.axvline(x=0.7, linestyle='--')
    _ = plt.tight_layout()
    _ = fig.savefig(f'Images/LinReg/LinReg_Error_Analysis_{metric}.png', bbox_inches='tight')


def really_check_single(ALL_DATA, iters=100, prop=0.7, plot=False):
    # Stick with train_proportion=0.7
    training_results, testing_results = [], []
    for sim in range(iters):
        ALL_DATA = shuffle_data(ALL_DATA)
        TRAIN_DF, TEST_DF = split_data(ALL_DATA, TRAIN_PROPORTION=prop)
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


# training_results, testing_results = load_check_single_70()
#
#
# def visualize_check_single(training_results, testing_results):
#     # _ = plt.hist(testing_results[:, 2], bins=20)
#     pass


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
scaler.fit(arr)
ALL_DATA = pd.DataFrame(scaler.transform(arr))

_ = """
################################################################################
################################## SINGLE-RUN ##################################
################################################################################
"""

# Split Data
TRAIN_DF, TEST_DF = split_data(ALL_DATA, 0.7)
# Train model
model = init_model()
r_train, r_sq_train, std_err_train, rmse_train, _ = train_model(model, TRAIN_DF, plot=True)
r_test, r_sq_test, std_err_test, rmse_test, _ = test_model(model, TEST_DF, plot=True)

_ = """
################################################################################
############################ SPLIT SIZE VS. METRICS ############################
################################################################################
"""
# # Conduct Simulation
(training_tracker, test_tracker, training_std_err,
 testing_std_err, training_rmse, testing_rmse) = run_many_splits(ALL_DATA)
save_data(training_tracker, test_tracker, training_std_err,
          testing_std_err, training_rmse, testing_rmse)
# Load Already-Conducted Simulation Results
training_tracker, test_tracker, training_std_err, testing_std_err, training_rmse, testing_rmse = load_data()
# Plot errors and fit
plt.plot(training_rmse.keys(), training_rmse.values())
plt.plot(testing_rmse.keys(), testing_rmse.values())
error_analysis(training_tracker, test_tracker, training_std_err, testing_std_err, metric='Std Error')
error_analysis(training_tracker, test_tracker, training_rmse, testing_rmse, metric='RMSE')

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

# TRAIN_DF, TEST_DF = split_data(ALL_DATA, TRAIN_PROPORTION=0.7)
arr = ALL_DATA.values
design_matrix = arr[:, :-1]
prediction_column = arr[:, -1]
# Train model
rmse = []
for k in range(2, 30):
    model = init_model()
    cv = KFold(n_splits=k, random_state=1, shuffle=True)
    scores = cross_val_score(model, design_matrix, prediction_column, scoring='neg_mean_squared_error',
                             cv=cv, n_jobs=-1)
    rmse.append(np.mean(scores) * -1)

_ = plt.xlabel('Fold')
_ = plt.ylabel('RMSE')
_ = plt.plot(rmse)
# EOF
