# @Author: shounak
# @Date:   2022-11-25T19:11:59-08:00
# @Email:  shounak@stanford.edu
# @Filename: utll.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-26T15:25:12-08:00


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import sklearn

_ = """
################################################################################
############################# FOR MACHINE LEARNING #############################
################################################################################
"""


def shuffle_data(df):
    return df.sample(frac=1).reset_index(drop=True)


def draw_line(slope, intercept, x_tremas):
    """Plot a line from slope and intercept"""
    # https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
    # axes = plt.gca()
    x_vals = np.array(x_tremas)
    y_vals = intercept + slope * x_vals
    _ = plt.plot(x_vals, y_vals, '--', color='orange', linewidth=3)


def _feed_through_model(model, design_matrix, prediction_column, plot, dataset_type):
    guesses = model.predict(design_matrix)
    rmse = sklearn.metrics.mean_squared_error(prediction_column, guesses)
    mape = sklearn.metrics.mean_absolute_percentage_error(prediction_column, guesses)
    if plot:
        _ = plt.figure(figsize=(15, 15))
        # Visualize Performance on Test Dataset
        _ = plt.title(f'{dataset_type} data')
        _ = plt.ylabel('Predicted T_Complication')
        _ = plt.scatter(prediction_column, guesses)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(guesses, prediction_column)
    x_min, x_max = prediction_column.min(), prediction_column.max()
    if plot:
        draw_line(slope, intercept, x_tremas=(x_min, x_max))
        caption = f"r: {round(r_value, 3)}, r-sq: {round(r_value**2, 3)}"
        _ = plt.title(f'{dataset_type} data')
        _ = plt.xlabel('True T_Complication\n' + caption)
        _ = plt.tight_layout()
        _ = plt.savefig(f'Images/LinReg/LinReg_on_{dataset_type}.png', bbox_inches='tight')
        _ = plt.show()

    return r_value, r_value**2, std_err, rmse, mape


def split_data(ALL_DATA, TRAIN_PROPORTION):
    TRAIN_SIZE = int(len(ALL_DATA) * TRAIN_PROPORTION)
    TRAIN_DF = ALL_DATA.values[:TRAIN_SIZE, :]
    TEST_DF = ALL_DATA.values[TRAIN_SIZE:, :]
    return TRAIN_DF, TEST_DF


_ = """
################################################################################
################################ FOR PROCESSING ################################
################################################################################
"""


def cat_to_num(series):
    return series.astype('category').cat.codes


def remove_highly_correlated(df, THRESHOLD=0.5, skip=[], plot=False):
    corr_matrix = df.corr().abs()
    # Exploratory
    if plot:
        _ = sns.heatmap(corr_matrix)
        _ = plt.hist(corr_matrix.values.flatten(), bins=20)
        _ = plt.show()
    # np.quantile(correlation_flat, 0.95)
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [col for col in upper.columns if any(upper[col] > THRESHOLD) if col not in skip]
    perc = round(100 * len(to_drop) / len(list(df)), 3)
    print(f'Removing {len(to_drop)} (~{perc}%) feature(s), highly-correlated.\n{to_drop}')
    return df.drop(to_drop, axis=1)


def nan_cols(df):
    # Sanity Check
    return {c for c, v in df.isna().any().to_dict().items() if v is True}


def assign_age_complication(row, VAL_FOR_NO_COMPLICATION_YET):
    # If the person had a non-death complication, keep it
    if row['Had_Complication']:
        assert row['Age_of_Complication'] != VAL_FOR_NO_COMPLICATION_YET
        return row['Age_of_Complication']
    else:
        if row['Deceased']:
            # If the person did not have a complication but died, their death IS the complication
            return row['Age at Death']
        else:
            # If the person did not have a complication and is still alive, keep dummy value
            # These are the unlabeled instances for the semi-supervised learning
            assert row['Age_of_Complication'] == VAL_FOR_NO_COMPLICATION_YET
            return row['Age_of_Complication']


def non_sparse_columns(df, THRESHOLD):
    # num_nas = {}
    # for col in df.columns:
    #     num_nas[col] = df[col].isna().sum() / len(df[col])
    # num_nas = {col: value for col, value in num_nas.items() if value <= THRESHOLD}
    # # np.quantile(list(num_nas.values()), 0.5)
    # # plt.hist(list(num_nas.values()), bins=20)
    # df = df[list(num_nas.keys())].fillna(0.)
    # return df
    to_survive = int((1 - THRESHOLD) * len(df))
    return df.dropna(axis=1, thresh=to_survive).fillna(0.)


def save_data(data, path):
    data.to_pickle(path + '.pkl')
    data.to_csv(path + '.csv')

# EOF
