# @Author: shounak
# @Date:   2022-11-22T23:18:49-08:00
# @Email:  shounak@stanford.edu
# @Filename: models.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-24T01:55:21-08:00

import pandas as pd
from sklearn.linear_model import LinearRegression


_ = """
################################################################################
############################# FUNCTION DEFINITIONS #############################
################################################################################
"""


def shuffle_data(df):
    return df.sample(frac=1).reset_index(drop=True)


_ = """
################################################################################
############################# SOME HYPER-PARAMETERS ############################
################################################################################

"""
TRAIN_SIZE = 0.7

_ = """
################################################################################
########################## LINEAR REGRESSION TRAINING ##########################
################################################################################

"""

ALL_DATA = pd.read_csv('Data/Merged Complete/Example_Core_Dataset.csv').infer_objects()
# ALL_DATA.drop('Patient Id', axis=1, inplace=True)
TRAIN_SIZE = int(len(ALL_DATA) * TRAIN_SIZE)
{c for c, v in ALL_DATA.isna().any().to_dict().items() if v is True}
ALL_DATA['Notes'].unique()

# Just drop Notes
# Time Until Complication for dead people should be `Age at Death` - `Transplant Date`

shuffled = shuffle_data(ALL_DATA)
TRAIN_DF = shuffled.values[:TRAIN_SIZE, :]
TEST_DF = shuffled.values[TRAIN_SIZE:, :]

model = LinearRegression()
design_matrix = TRAIN_DF[:, :-1]
prediction_column = TRAIN_DF[:, -1]
model.fit(design_matrix, prediction_column)
model.score(design_matrix, prediction_column)

test_design_matrix = TEST_DF[:, :-1]
test_prediction_column = TEST_DF[:, -1]
model.predict()
