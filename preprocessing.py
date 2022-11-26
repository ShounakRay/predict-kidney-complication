# @Author: shounak
# @Date:   2022-10-28T11:07:35-07:00
# @Email:  shounak@stanford.edu
# @Filename: preprocessing.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-25T20:41:28-08:00

import pandas as pd

import numpy as np
from datetime import datetime

from util import (cat_to_num, remove_highly_correlated, save_data,
                  nan_cols, assign_age_complication, non_sparse_columns)

FPATHS_FULL = {
    'Demographics': 'Data/Complete Constituents/all_demographics.csv',
    'Diagnoses': 'Data/Complete Constituents/all_diagnoses.csv',
    'Labs': 'Data/Complete Constituents/all_labs.csv',
    'MedOrders': 'Data/Complete Constituents/all_medorders.csv',
    'Procedures': 'Data/Complete Constituents/all_procedures.csv'}

SAVE_PATH_FULL = 'Data/Merged Complete/Core_Dataset'

data = {}
for label, path in FPATHS_FULL.items():
    data[label] = pd.read_csv(path, low_memory=False).infer_objects()
    print(f'Ingested "{label}" dataset.')
COPY_DATA = data.copy()

# UNUSED:
# data['Codebook'] maps patient names to MRN code
# data['Notes'] has patient clininal meeting notes

_ = """
################################################################################
############################# DATA HYPERPARAMETERS #############################
################################################################################
"""
# Semi-supervised context: Used, but no ultimate bearing on dataset
# Supervised context: Represents `Age_of_Complication` for people that never had a complication
#                     Implicitly impacts `Time_Until_Complication` (= Age_of_Complication - Age_of_Transplant)
VAL_FOR_NO_COMPLICATION_YET = 100
# What is the highest PERCENTAGE of NaN values you're comfortable with per column? Used in Diagnoses and Procedures Dataset
# TODO: Why not in demographics or medication orders?
DIAG_NAN_MOST = 0.85
# # What is the highest NUMBER of NaN values you're comfortable with per column? Used in Labs
# MAX_NUM_NAN_LABS = 1150

_ = """
################################################################################
############################# FUNCTION DEFINITIONS #############################
################################################################################
"""


def medication_transformation(df):
    # Definitely remove some things
    intially_remove_these = ['Sig', 'Route', 'Disp', 'Unit', 'Refills',
                             'Frequency', 'Number of Times', 'Order Status',
                             'Order Class', 'Order Mode', 'Prescribing Provider',
                             'PatEncCsnCoded', 'Order Date', 'Order Age']
    df.drop(intially_remove_these, axis=1, inplace=True)

    # Engineer duration of consumption
    df['End Date'] = pd.to_datetime(df['End Date'])
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Med_Duration_Days'] = (df['End Date'] - df['Start Date']).dt.days

    # Remove some more things
    remove_finally = ['Start Date', 'End Date', 'End Age']
    df.drop(remove_finally, axis=1, inplace=True)

    # There are many fewer therapeutic classes than Pharmaceutical classes. So, we're going
    # to keep the therapeutic classes.
    df['Medication'] = df['Medication'].str.upper()
    df['Therapeutic Class'] = df['Therapeutic Class'].str.upper()
    drop_experimental = ['Medication', 'Ingredients', 'Pharmaceutical Class']
    df.drop(drop_experimental, axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df[df['Med_Duration_Days'] >= 0].reset_index(drop=True)

    # Per patient, generate count of each therapeutic class
    df['count'] = 1
    avg_days_per_class = df.groupby(['Patient Id', 'Therapeutic Class']).agg('mean')
    avg_days_per_class = avg_days_per_class.pivot_table('Med_Duration_Days', ['Patient Id'], 'Therapeutic Class')
    avg_days_per_class.columns = 'AvgDaysOn_' + avg_days_per_class.columns
    avg_days_per_class = avg_days_per_class.dropna(axis=1, how='all').fillna(0.)
    avg_days_per_class.reset_index(inplace=True)

    return avg_days_per_class   # NOTE: THIS IS A PERFECT DATASET


def demographics_transformation(df):
    # Definitely remove some things
    to_remove_demographics = ['Disposition', 'Marital Status', 'Interpreter Needed',
                              'Insurance Name', 'Insurance Type', 'Zipcode',
                              'Death Date SSA Do Not Disclose', 'Comment',
                              'City', 'Tags', 'Smoking History Taken',
                              'Date of Death']
    df.drop(to_remove_demographics, axis=1, inplace=True)
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])
    df['Current_Age'] = (datetime.now() - df['Date of Birth']).dt.days / 365.25

    # Encoding some demographics
    df['Gender'] = cat_to_num(df['Gender'])
    df['Deceased'] = cat_to_num(df['Deceased'])
    df['Race'] = cat_to_num(df['Race'])
    df['Ethnicity'] = cat_to_num(df['Ethnicity'])

    # Manual dtype correction + Sanitation
    df['Smoking Hx'] = df['Smoking Hx'].str[0].astype(np.number)
    # Sanitation: set current ages of people that are deceased to their age at death
    df['Current_Age'] = df.apply(lambda row: row['Current_Age']
                                 if row['Deceased'] == 0 else row['Age at Death'],
                                 axis=1)
    df = df.select_dtypes(include=np.number)

    """
    {'Age at Death',        # Will use this, and then delete
     'Notes',               # Will delete right now (SEE below)
     'Recent BMI',          # Will keep, need to handle (imputed, SEE below)
     'Recent Height cm',    # Will be removed (SEE below)
     'Recent Weight kg',    # Will be removed (SEE below)
     'Smoking Hx'           # Need to investigate (imputed, SEE below)
     }
    """
    # Manually removing this correlation with BMI
    df.drop(['Recent Height cm', 'Recent Weight kg'], axis=1, inplace=True)
    df.drop(['Notes'], axis=1, inplace=True)
    # ASSUMPTION: Imputation
    df['Smoking Hx'] = df['Smoking Hx'].fillna(
        df['Smoking Hx'].mean())
    df['Recent BMI'] = df['Recent BMI'].fillna(
        df['Recent BMI'].mean())

    return df   # NOTE: THIS IS A PERFECT DATASET


def diagnoses_transformation(df):
    # Definitely remove some things
    to_remove_diagnoses = ['Source', 'ICD9 Code', 'Billing Provider',
                           'PatEncCsnCoded', 'Type', 'Description', 'Performing Provider']
    df.drop(to_remove_diagnoses, axis=1, inplace=True)
    # The age associated with a diagnosis is not necessarily the same as the current age of patient
    # Just a trivial change, for readability
    df['Age_Of_Diagnosis'] = df['Age']

    """SEGMENT 1: Important; engineering TERM for our prediction_column, `Time_Until_Complication` (if at all) here"""
    # T86.11 is the code for kidney transplant rejection. Only get those specific diagnoses.
    ppl_final_rejection = df[df['ICD10 Code'] == 'T86.11'].reset_index(drop=True)
    ppl_final_rejection.drop_duplicates(subset=['Patient Id'], keep='first', inplace=True, ignore_index=True)
    TIME_COMPLICATION_MAPPING = ppl_final_rejection.set_index('Patient Id')['Age_Of_Diagnosis'].to_dict()

    """SEGMENT 2: Pivoting the table"""
    temp = df.copy()

    # Per patient, generate count of each ICD10 Code
    temp['count'] = 1
    temp['ICD10 Code'] = temp['ICD10 Code'].astype(str)
    # Remove all the Z (Factors influencing health status and contact with health services) codes
    temp = temp[~temp['ICD10 Code'].str.startswith('Z')]
    track_code_freq = temp.groupby(['Patient Id', 'ICD10 Code']).agg({'count': 'count'})
    track_code_freq = track_code_freq.pivot_table('count', ['Patient Id'], 'ICD10 Code')

    # Remove columns that don't matter too much
    # The DIAG_NAN_MOST is a hyper-parameter
    temp = non_sparse_columns(track_code_freq, THRESHOLD=DIAG_NAN_MOST)

    temp.columns = 'CountOf_' + temp.columns
    temp.reset_index(inplace=True)

    """SEGMENT 3: Creating `Age_of_Complication` from `TIME_COMPLICATION_MAPPING` in SEGMENT 1"""
    # Now add the column we engineered:
    temp['Age_of_Complication'] = temp['Patient Id'].apply(
        lambda x: TIME_COMPLICATION_MAPPING.get(x, VAL_FOR_NO_COMPLICATION_YET))
    temp['Had_Complication'] = temp['Age_of_Complication'].apply(lambda x: not (x == VAL_FOR_NO_COMPLICATION_YET))

    return temp    # NOTE: THIS IS A PERFECT DATASET


def labs_transformation(df):
    # Definitely remove some things
    to_remove_labs = ['Order Date',
                      'Result Date',
                      'Authorizing Provider',
                      'PatEncCsnCoded']
    df.drop(to_remove_labs, axis=1, inplace=True)
    temp = df.copy()
    temp = temp.dropna(subset=['Abnormal'])
    temp['Age_Result'] = temp['Age']
    temp = temp[['Patient Id', 'Lab', 'Abnormal', 'Age_Result']]
    temp['result_count'] = 1
    lab_aggregations = temp.groupby(['Patient Id', 'Lab', 'Abnormal']).agg({'Age_Result': np.mean,
                                                                            'result_count': np.sum})
    # Pivot and breaks the multi-index hierarchy
    encoded_table = lab_aggregations.pivot_table('result_count', ['Patient Id'], ['Lab', 'Abnormal']).fillna(0)
    encoded_table.columns = encoded_table.columns.map(' >> '.join).str.strip(' >> ')

    # results = {col: (encoded_table[col] == 0.0).sum() for col in encoded_table.columns}
    # # HYPER-PARAMETER USAGE
    # results = {col: value for col, value in results.items() if value <= MAX_NUM_NAN_LABS}
    # most_common_columns = list(results.keys())
    # encoded_table = encoded_table[most_common_columns]
    encoded_table = non_sparse_columns(encoded_table, THRESHOLD=DIAG_NAN_MOST)
    # VALUE IN THE TABLE REPRESENTS NUMBER OF OCCURENCES

    return encoded_table


def procedures_transformation(df):
    TRANSPLANT_CODES = ['0TY10Z0', '0TY00Z0', '55.69']
    # Only get transplant procedures
    transplant_procedures = df[df['Code'].isin(TRANSPLANT_CODES)].drop_duplicates(
        subset=['Patient Id'], keep='first').reset_index(drop=True)
    AGE_OF_TRANSPLANT_MAPPING = transplant_procedures.set_index('Patient Id')['Age'].to_dict()
    df = df[['Patient Id', 'Age', 'Code', 'Description']]

    """ SECTION 2 """
    df[df['Description'].str.contains('transplant') & df['Description'].str.contains('kidney')]['Code'].unique()
    df['count'] = 1
    track_code_freq = df.groupby(['Patient Id', 'Code']).agg({'count': 'count'})
    track_code_freq = track_code_freq.pivot_table('count', ['Patient Id'], 'Code')

    """ SECTION 3 """
    df = non_sparse_columns(track_code_freq, THRESHOLD=DIAG_NAN_MOST)
    df.columns = 'CountOf_' + df.columns
    df.reset_index(inplace=True)
    df['Age_of_Transplant'] = df['Patient Id'].apply(lambda x: AGE_OF_TRANSPLANT_MAPPING.get(x, np.nan))
    df = df.dropna(subset=['Age_of_Transplant'], how='any')

    return df


_ = """
################################################################################
############################# DATA PRE-PROCESSING ##############################
################################################################################
"""

data['MedOrders'] = medication_transformation(data['MedOrders'])
# nan_cols(data['MedOrders'])
print("Completed Transformation for MEDORDERS.")

data['Demographics'] = demographics_transformation(data['Demographics'])
# nan_cols(data['Demographics'])
print("Completed Transformation for DEMOGRAPHICS.")

data['Diagnoses'] = diagnoses_transformation(data['Diagnoses'])
# nan_cols(data['Diagnoses'])
print("Completed Transformation for DIAGNOSES.")

data['Labs'] = labs_transformation(data['Labs'])
# Don't expect anything anyways in this case, all the NaNs were set to 0
# nan_cols(data['Labs'])
print("Completed Transformation for LABS.")

data['Procedures'] = procedures_transformation(data['Procedures'])
# Don't expect anything anyways in this case, all the NaNs were set to 0
# nan_cols(data['Procedures'])
print("Completed Transformation for PROCEDURES.")

_ = """
################################################################################
################# LAST-PASS: REMOVE HIGHLY CORRELATED VARIABLES ################
################################################################################
"""

KEEP_NO_MATTER_WHAT = ['Age_of_Complication', 'Age_of_Transplant', 'Current_Age',
                       'Deceased', 'Time_Until_Complication', 'Had_Complication']

data['MedOrders'] = remove_highly_correlated(data['MedOrders'], skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.5)
data['Demographics'] = remove_highly_correlated(data['Demographics'], skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.5)
data['Diagnoses'] = remove_highly_correlated(data['Diagnoses'], skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.5)
data['Labs'] = remove_highly_correlated(data['Labs'], THRESHOLD=0.7, skip=KEEP_NO_MATTER_WHAT)
data['Procedures'] = remove_highly_correlated(data['Procedures'], skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.5)
print("Completed INDEPENDENT removal of highly cross-correlated features.")

_ = """
################################################################################
##################################### MERGE ####################################
################################################################################
"""

NATURE_OF_JOIN = {'how': 'inner',
                  'on': ['Patient Id'],
                  'lsuffix': '_DELETE'}

"""MERGE MEDICAL HISTORY + DEMOGRAPHICS"""
merged_one = data['MedOrders'].join(data['Demographics'], rsuffix='_demo', **NATURE_OF_JOIN)
merged_one.drop('Patient Id_DELETE', axis=1, inplace=True)

""" MERGED FIRST_MERGED + DIAGNOSES """
merged_two = merged_one.join(data['Diagnoses'], rsuffix='_diagnosis', **NATURE_OF_JOIN)
merged_two.drop('Patient Id_DELETE', axis=1, inplace=True)

""" MERGED SECOND_MERGED + LABS """
merged_three = merged_two.join(data['Labs'], rsuffix='_labs', **NATURE_OF_JOIN)

""" MERGED THIRD_MERGED + PROCEDURES """
merged_four = merged_three.join(data['Procedures'], rsuffix='_procedures', **NATURE_OF_JOIN)
print("Completed merging all datasets together.")

_ = """
################################################################################
########################## FINAL, POST-MERGE DELETIONS #########################
################################################################################
"""
merged_four.drop(['Patient Id_demo', 'CountOf_nan', 'Patient Id_procedures',
                  'Patient Id_diagnosis', 'Patient Id_DELETE'], axis=1, inplace=True)

# CORRECT/ASSIGN `Age_of_Complication`
merged_four['Age_of_Complication'] = merged_four.apply(
    assign_age_complication, args=(VAL_FOR_NO_COMPLICATION_YET), axis=1)
# CALCULATE OUR PREDICTION VARIABLE: Time_Until_Complication
merged_four['Time_Until_Complication'] = merged_four['Age_of_Complication'] - merged_four['Age_of_Transplant']
merged_four = merged_four[merged_four['Time_Until_Complication'] >= 0].reset_index(drop=True)

# Final pass to collectively analyse all columns and remove high correlations
FINAL_UNCORR = remove_highly_correlated(merged_four, skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.7)
# REORDER COLUMNS SO `Time_Until_Complication` IS AT THE VERY END
new_order = [c for c in list(FINAL_UNCORR) if c not in ['Patient Id',
                                                        'Time_Until_Complication']] + ['Time_Until_Complication']
FINAL_UNCORR = FINAL_UNCORR[new_order]
print("Completed last-pass construction of prediction variable + final correlation check.")

_ = """
################################################################################
################################### SAVE FILE ##################################
################################################################################
"""

""" LOGIC FOR SUPERVISED, IMPUTED VALUES """
save_data(FINAL_UNCORR.drop(['Age_of_Complication', 'Age_of_Transplant', 'Age at Death', 'Had_Complication'],
                            axis=1, inplace=False),
          path=SAVE_PATH_FULL + '_SUPERVISED')

""" LOGIC FOR SEMI - SUPERVISED, IMPUTED VALUES """
# Can't depend on Had_Complication for final None assignment (just look for `VAL_FOR_NO_COMPLICATION_YET` value)
FINAL_UNCORR['Time_Until_Complication'] = FINAL_UNCORR['Age_of_Complication'].apply(
    lambda x: np.nan if x == VAL_FOR_NO_COMPLICATION_YET else x)
save_data(FINAL_UNCORR.drop(['Age_of_Complication', 'Age_of_Transplant', 'Age at Death', 'Had_Complication'],
                            axis=1, inplace=False),
          path=SAVE_PATH_FULL + '_SEMI-SUPERVISED')
# Number of Unlabeled instances 0.68 = FINAL_UNCORR['Time_Until_Complication'].isna().sum() / len(FINAL_UNCORR['Time_Until_Complication'])
# EOF
