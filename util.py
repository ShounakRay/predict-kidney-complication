# @Author: shounak
# @Date:   2022-11-25T19:11:59-08:00
# @Email:  shounak@stanford.edu
# @Filename: utll.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-26T00:30:16-08:00


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

_ = """
################################################################################
############################# FOR MACHINE LEARNING #############################
################################################################################
"""


def shuffle_data(df):
    return df.sample(frac=1).reset_index(drop=True)


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
        plt.show()
    # np.quantile(correlation_flat, 0.95)
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [col for col in upper.columns if any(upper[col] > THRESHOLD) if col not in skip]
    perc = len(to_drop) / len(list(df))
    print(f'Removing {len(to_drop)} (perc) feature(s), highly-correlated.\n{to_drop}')
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
