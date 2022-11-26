# @Author: shounak
# @Date:   2022-10-28T11:07:35-07:00
# @Email:  shounak@stanford.edu
# @Filename: preprocessing.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-26T00:34:40-08:00

import pandas as pd
import matplotlib.pyplot as plt
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
# data = COPY_DATA.copy()
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
# TODO: Why not in demographics orders?
DIAG_NAN_MOST = 0.85
# # What is the highest NUMBER of NaN values you're comfortable with per column? Used in Labs
# MAX_NUM_NAN_LABS = 1150
# TODO: Should track lab/procedure/diagnosis count for all days LEADING UP TO COMPLICATION (TRANSPLANT or DEATH)
# TODO: Counts should be based on after complication, not for lifetime.

_ = """
################################################################################
############################# FUNCTION DEFINITIONS #############################
################################################################################
"""


def diagnoses_transformation(data):
    df = data['Diagnoses'].copy()
    # Definitely remove some things
    to_remove_diagnoses = ['Source', 'ICD9 Code', 'Billing Provider',
                           'PatEncCsnCoded', 'Type', 'Description', 'Performing Provider']
    df.drop(to_remove_diagnoses, axis=1, inplace=True)
    # The age associated with a diagnosis is not necessarily the same as the current age of patient
    # Just a trivial change, for readability
    df['Age_Of_Diagnosis'] = df['Age']
    df.drop('Age', axis=1, inplace=True)

    """SEGMENT 1: Important; engineering TERM for our prediction_column, `Time_Until_Complication` (if at all) here"""
    # T86.11 is the code for kidney transplant rejection. Only get those specific diagnoses.
    ppl_final_rejection = df[df['ICD10 Code'] == 'T86.11'].reset_index(drop=True)
    ppl_final_rejection.drop_duplicates(subset=['Patient Id'], keep='first', inplace=True, ignore_index=True)
    TIME_COMPLICATION_MAPPING = ppl_final_rejection.set_index('Patient Id')['Age_Of_Diagnosis'].to_dict()
    # Can't populate all the keys yet since we don't know who died
    DATE_COMPLICATION_MAPPING = pd.to_datetime(ppl_final_rejection.set_index('Patient Id')['Date']).to_dict()

    """Immediately get rid of rows not required"""
    demographics_data, DATE_COMPLICATION_MAPPING = update_demo_and_datemapping(
        data['Demographics'].copy(), DATE_COMPLICATION_MAPPING)
    data['Demographics'] = demographics_data
    df['Date'] = pd.to_datetime(df['Date'])
    df = constrain_to_complication_date(df.copy(), DATE_COMPLICATION_MAPPING, date_col='Date')

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

    data['Diagnoses'] = temp

    return data, DATE_COMPLICATION_MAPPING    # NOTE: THIS IS A PERFECT DATASET


def medication_transformation(df, DATE_COMPLICATION_MAPPING):
    # TODO: Optimization, takes a lot of time
    df['End Date'] = pd.to_datetime(df['End Date'])
    df['Start Date'] = pd.to_datetime(df['Start Date'])

    # Immediately remove useless rows
    df = constrain_to_complication_date(df.copy(), DATE_COMPLICATION_MAPPING, date_col='Start Date')

    # Definitely remove some things
    intially_remove_these = ['Sig', 'Route', 'Disp', 'Unit', 'Refills',
                             'Frequency', 'Number of Times', 'Order Status',
                             'Order Class', 'Order Mode', 'Prescribing Provider',
                             'PatEncCsnCoded', 'Order Date', 'Order Age']
    df.drop(intially_remove_these, axis=1, inplace=True)

    # Engineer duration of consumption
    df['Med_Duration_Days'] = (df['End Date'] - df['Start Date']).dt.days

    # Remove some more things
    remove_finally = ['Start Date', 'End Date', 'End Age']
    df.drop(remove_finally, axis=1, inplace=True)

    # There are many fewer therapeutic classes than Pharmaceutical classes. So, we're going
    # to keep the therapeutic classes.
    drop_experimental = ['Medication', 'Ingredients', 'Pharmaceutical Class']
    df.drop(drop_experimental, axis=1, inplace=True)
    # df.dropna(axis=1, inplace=True)   WE DON'T ACTUALLY WANT TO DROP ROWS JUST YET
    df = df[df['Med_Duration_Days'] >= 0].reset_index(drop=True)

    # Per patient, generate count of each therapeutic class
    df['count'] = 1
    df['Therapeutic Class'] = df['Therapeutic Class'].str.upper()
    avg_days_per_class = df.groupby(['Patient Id', 'Therapeutic Class']).agg('mean')
    avg_days_per_class = avg_days_per_class.pivot_table('Med_Duration_Days', ['Patient Id'], 'Therapeutic Class')
    avg_days_per_class.columns = 'AvgDaysOn_' + avg_days_per_class.columns
    # avg_days_per_class.dropna(axis=1, how='all', inplace=True)
    avg_days_per_class.reset_index(inplace=True)

    avg_days_per_class = non_sparse_columns(avg_days_per_class.copy(), THRESHOLD=DIAG_NAN_MOST)
    # avg_days_per_class.dropna(axis=1, thresh=int(0.15 * len(avg_days_per_class)))

    return avg_days_per_class   # NOTE: THIS IS A PERFECT DATASET


def labs_transformation(df, DATE_COMPLICATION_MAPPING):
    # Immediately remove useless rows
    df['Taken Date'] = pd.to_datetime(df['Taken Date'])
    df = constrain_to_complication_date(df.copy(), DATE_COMPLICATION_MAPPING, date_col='Taken Date')

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
    encoded_table = lab_aggregations.pivot_table('result_count', ['Patient Id'], ['Lab', 'Abnormal'])
    encoded_table.columns = encoded_table.columns.map(' >> '.join).str.strip(' >> ')

    # results = {col: (encoded_table[col] == 0.0).sum() for col in encoded_table.columns}
    # # HYPER-PARAMETER USAGE
    # results = {col: value for col, value in results.items() if value <= MAX_NUM_NAN_LABS}
    # most_common_columns = list(results.keys())
    # encoded_table = encoded_table[most_common_columns]
    encoded_table = non_sparse_columns(encoded_table, THRESHOLD=DIAG_NAN_MOST)
    # VALUE IN THE TABLE REPRESENTS NUMBER OF OCCURENCES

    return encoded_table


def procedures_transformation(df, DATE_COMPLICATION_MAPPING):
    # Immediately constrain
    df['Date'] = pd.to_datetime(df['Date'])
    df = constrain_to_complication_date(df.copy(), DATE_COMPLICATION_MAPPING, date_col='Date')

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


def demographics_transformation(df):
    # Definitely remove some things
    to_remove_demographics = ['Disposition', 'Marital Status', 'Interpreter Needed',
                              'Insurance Name', 'Insurance Type', 'Zipcode',
                              'Death Date SSA Do Not Disclose', 'Comment',
                              'City', 'Tags', 'Smoking History Taken']
    # 'Date of Death']
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
    # df = df.select_dtypes(include=np.number)
    # Sanitation: set current ages of people that are deceased to their age at death
    df['Current_Age'] = df.apply(lambda row: row['Current_Age']
                                 if row['Deceased'] == 0 else row['Age at Death'],
                                 axis=1)

    # BIG NOTE: Upon investigation, we discovered that we don't know when 81 dead people died.
    #           So, we can't populate their "Current Age" value. So, we could impute these values
    #           (TODO), but we're choosing to delete these 81 cases (2% of all the data)
    # data['Demographics'][data['Demographics']['Age at Death'].isna()]
    # data['Demographics']['Date of Death'].isna().sum()
    # data['Demographics']['Age at Death'].isna().sum()
    # nan_cols(df)

    # Delete cases when we don't know when dead people died
    df.dropna(subset=['Current_Age'], how='any', inplace=True)

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

    return df.reset_index(drop=True)   # NOTE: THIS IS A PERFECT DATASET


def constrain_to_complication_date(df_full, DATE_COMPLICATION_MAPPING, date_col='Start Date'):
    df_full['DUPL_Patient Id'] = df_full['Patient Id']
    # Immediately remove all history for a patient before their date of complication
    # df.groupby('Patient Id').apply(
    #     lambda group: group[group['Start Date'] <= DATE_COMPLICATION_MAPPING.get(group['DUPL_Patient Id'].unique()[0])])
    dfs = []
    for patient_id, group in df_full.groupby('Patient Id'):
        # TODO: 31 patients where backup is used, why?
        date_complication = DATE_COMPLICATION_MAPPING.get(patient_id, pd.to_datetime(datetime.now()))
        dfs.append(group[group[date_col] <= date_complication])
    df_full = pd.concat(dfs).reset_index(drop=True)
    print("> Completed redundancy check")
    df_full.drop('DUPL_Patient Id', axis=1, inplace=True)
    return df_full


def update_demo_and_datemapping(demographics_data, DATE_COMPLICATION_MAPPING):
    """PROCESS DEMOGRAPHICS"""
    # Need to find out who's dead so we can complete `DATE_COMPLICATION_MAPPING`
    demographics_data = demographics_transformation(demographics_data.copy())
    assert nan_cols(demographics_data) == {'Age at Death', 'Alcohol Use', 'Date of Death', 'Language', 'Occupation',
                                           'Recent Encounter Date', 'Smoking Quit', 'Smoking Started'}
    print(f"Completed Transformation for DEMOGRAPHICS. Size: {demographics_data.shape}")

    """############################ DEATH THINGS ####################################"""
    death_dates = pd.to_datetime(demographics_data.set_index('Patient Id')['Age at Death']).dropna().to_dict()
    all_patients = demographics_data['Patient Id'].unique()
    for patient_id in all_patients:
        # If the patient didn't die, they could have natural complication date
        # (or NOT, they never had a true complication and they're still alive)
        if patient_id in death_dates.keys():
            # If the patient died, they must have a death date
            # They're dead. If they didn't already have a TRUE complication (not in MAPPING), then assign the death date.
            #   AKA if they're in DATE_COMPLICATION_MAPPING, they had a TRUE complication
            if patient_id not in DATE_COMPLICATION_MAPPING.keys():
                assert DATE_COMPLICATION_MAPPING.get(patient_id) is None    # For the patient that died
                death_date_patient = death_dates.get(patient_id)
                assert death_date_patient is not None
                DATE_COMPLICATION_MAPPING[patient_id] = pd.to_datetime(death_date_patient)
        else:
            # HYPER-PARAMETER-ISH
            DATE_COMPLICATION_MAPPING[patient_id] = datetime.now()

    # Consistent datatypes
    DATE_COMPLICATION_MAPPING = {k: pd.to_datetime(v) for k, v in DATE_COMPLICATION_MAPPING.items()}

    # Delete useless demographics
    demographics_data = demographics_data.select_dtypes(np.number)
    """################################ END ########################################"""

    return demographics_data, DATE_COMPLICATION_MAPPING


_ = """
################################################################################
############################# DATA PRE-PROCESSING ##############################
################################################################################
"""
# NOTE: ORDER MATTERS!!

# data['Diagnoses'] = COPY_DATA['Diagnoses'].copy()
# NOTE: This function also processes demographics inside
data, DATE_COMPLICATION_MAPPING = diagnoses_transformation(data.copy())
assert len(nan_cols(data['Diagnoses'])) == 0
print(f"Completed Transformation for DIAGNOSES. Size: {data['Diagnoses'].shape}")

# df = data['MedOrders'].copy()
data['MedOrders'] = medication_transformation(data['MedOrders'].copy(), DATE_COMPLICATION_MAPPING)
assert len(nan_cols(data['MedOrders'])) == 0
print(f"\nCompleted Transformation for MEDORDERS. Size: {data['MedOrders'].shape}")

# data['Labs'] = data['Labs'].copy()
data['Labs'] = labs_transformation(data['Labs'].copy(), DATE_COMPLICATION_MAPPING)
# Don't expect anything anyways in this case, all the NaNs were set to 0
assert len(nan_cols(data['Labs'])) == 0
print(f"Completed Transformation for LABS. Size: {data['Labs'].shape}")

data['Procedures'] = procedures_transformation(data['Procedures'].copy(), DATE_COMPLICATION_MAPPING)
# Don't expect anything anyways in this case, all the NaNs were set to 0
# nan_cols(data['Procedures'])
print(f"Completed Transformation for PROCEDURES. Size: {data['Procedures'].shape}")


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
data['Labs'] = remove_highly_correlated(data['Labs'], THRESHOLD=0.5, skip=KEEP_NO_MATTER_WHAT)
data['Procedures'] = remove_highly_correlated(data['Procedures'], skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.5)
print("\nCompleted INDEPENDENT removal of highly cross-correlated features.")

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
print("\nCompleted merging all datasets together.")

_ = """
################################################################################
########################## FINAL, POST-MERGE DELETIONS #########################
################################################################################
"""
merged_four.drop(['Patient Id_demo', 'CountOf_nan', 'Patient Id_procedures',
                  'Patient Id_diagnosis', 'Patient Id_DELETE'], axis=1, inplace=True)

# CORRECT/ASSIGN `Age_of_Complication`
# Doing this here becuase we need the `Deceased Column`
merged_four['Age_of_Complication'] = merged_four.apply(
    assign_age_complication, args=[VAL_FOR_NO_COMPLICATION_YET], axis=1)
# CALCULATE OUR PREDICTION VARIABLE: Time_Until_Complication
merged_four['Time_Until_Complication'] = merged_four['Age_of_Complication'] - merged_four['Age_of_Transplant']
merged_four = merged_four[merged_four['Time_Until_Complication'] >= 0].reset_index(drop=True)

# Final pass to collectively analyse all columns and remove high correlations
FINAL_UNCORR = remove_highly_correlated(merged_four, skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.7)
# REORDER COLUMNS SO `Time_Until_Complication` IS AT THE VERY END
new_order = [c for c in list(FINAL_UNCORR) if c not in ['Patient Id',
                                                        'Time_Until_Complication']] + ['Time_Until_Complication']
FINAL_UNCORR = FINAL_UNCORR[new_order]
print("\nCompleted last-pass construction of prediction variable + final correlation check.")


_ = """
################################################################################
################################### SAVE FILE ##################################
################################################################################
"""

""" LOGIC FOR SUPERVISED, IMPUTED VALUES """
save_data(FINAL_UNCORR.drop(['Age_of_Complication', 'Age_of_Transplant', 'Age at Death', 'Had_Complication'],
                            axis=1, inplace=False),
          path=SAVE_PATH_FULL + '_SUPERVISED')
print("\nSaved supervised dataset.")

""" LOGIC FOR SEMI - SUPERVISED, IMPUTED VALUES """
# Can't depend on Had_Complication for final None assignment (just look for `VAL_FOR_NO_COMPLICATION_YET` value)
FINAL_UNCORR['Time_Until_Complication'] = FINAL_UNCORR['Age_of_Complication'].apply(
    lambda x: np.nan if x == VAL_FOR_NO_COMPLICATION_YET else x)
save_data(FINAL_UNCORR.drop(['Age_of_Complication', 'Age_of_Transplant', 'Age at Death', 'Had_Complication'],
                            axis=1, inplace=False),
          path=SAVE_PATH_FULL + '_SEMI-SUPERVISED')
print("\nSaved semi-supervised dataset.")
# Number of Unlabeled instances 0.68 = FINAL_UNCORR['Time_Until_Complication'].isna().sum() / len(FINAL_UNCORR['Time_Until_Complication'])
# EOF
