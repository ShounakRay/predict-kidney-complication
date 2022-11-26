# @Author: shounak
# @Date:   2022-11-22T23:18:49-08:00
# @Email:  shounak@stanford.edu
# @Filename: models.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-26T11:30:29-08:00

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy
from util import shuffle_data, nan_cols

# TODO: Implement StandardScaler for normalization
# TODO: Try out Sparse Regularization

_ = """
################################################################################
############################# SOME HYPER-PARAMETERS ############################
################################################################################
"""
TRAIN_PROPORTION = 0.7

_ = """
################################################################################
################################## MODEL WORK ##################################
################################################################################
"""
ALL_DATA = pd.read_csv('Data/Merged Complete/Core_Dataset_SUPERVISED.csv').infer_objects().drop('Unnamed: 0', axis=1)
# ALL_DATA.drop('Patient Id', axis=1, inplace=True)
TRAIN_SIZE = int(len(ALL_DATA) * TRAIN_PROPORTION)
assert len(nan_cols(ALL_DATA)) == 0

list(ALL_DATA)
axarr = ALL_DATA.hist(bins=20, figsize=(100, 100))
for ax in axarr.flatten():
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.savefig('Images/Data_Histogram.png')


# Create training and test data
shuffled = shuffle_data(ALL_DATA)
TRAIN_DF = shuffled.values[:TRAIN_SIZE, :]
TEST_DF = shuffled.values[TRAIN_SIZE:, :]

# Train Model
model = LinearRegression()
design_matrix = TRAIN_DF[:, :-1]
prediction_column = TRAIN_DF[:, -1]
model.fit(design_matrix, prediction_column)
# model.score(design_matrix, prediction_column) # Equiv to code below for getting r**2 value
# Plot on training
guesses = model.predict(design_matrix)
# Visualize Performance on Test Dataset
_ = plt.xlabel('TRAINING: True T_Complication')
_ = plt.ylabel('TRAINING: Predicted T_Complication')
_ = plt.scatter(prediction_column, guesses)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(guesses, prediction_column)
print(f"r: {r_value}")
print(f"r-sq: {r_value ** 2}")

# Test Model
test_design_matrix = TEST_DF[:, :-1]
test_prediction_column = TEST_DF[:, -1]
guesses = model.predict(test_design_matrix)
# Visualize Performance on Test Dataset
_ = plt.xlabel('TESTING: True T_Complication')
_ = plt.ylabel('TESTING: Predicted T_Complication')
_ = plt.scatter(test_prediction_column, guesses)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(guesses, test_prediction_column)
print(f"r: {r_value}")
print(f"r-sq: {r_value ** 2}")
