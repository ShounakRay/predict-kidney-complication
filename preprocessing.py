# @Author: shounak
# @Date:   2022-10-28T11:07:35-07:00
# @Email:  shounak@stanford.edu
# @Filename: preprocessing.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-25T18:58:16-08:00

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# FPATHS = {
#     # 'Notes': 'Data/Incomplete Constituents/kds1_clinical_note-002.csv',
#     'Demographics': 'Data/Incomplete Constituents/kds1_demographics.csv',
#     'Diagnoses': 'Data/Incomplete Constituents/kds1_diagnoses.csv',
#     'Labs': 'Data/Incomplete Constituents/kds1_labs-001.csv',
#     'MedOrders': 'Data/Incomplete Constituents/kds1_med_orders.csv',
#     # 'Codebook': 'Data/Incomplete Constituents/kds1_patientCodebook.csv',
#     'Procedures': 'Data/Incomplete Constituents/kds1_procedures.csv'}
#
# SAVE_PATH = 'Data/Merged Incomplete/Example_Core_Dataset'

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

# data['Codebook'] maps patient names to MRN code
# data['Notes'] has patient clininal meeting notes

# TODO: Age at Death

_ = """
################################################################################
############################# FUNCTION DEFINITIONS #############################
################################################################################
"""


def cat_to_num(series):
    return series.astype('category').cat.codes


def remove_highly_correlated(df, THRESHOLD=0.5, skip=[]):
    corr_matrix = df.corr().abs()
    # Exploratory
    _ = sns.heatmap(corr_matrix)
    _ = plt.hist(corr_matrix.values.flatten(), bins=20)
    # np.quantile(correlation_flat, 0.95)
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [col for col in upper.columns if any(upper[col] > THRESHOLD) if col not in skip]
    print(f'Removing {len(to_drop)} feature(s), highly-correlated.\n{to_drop}')
    return df.drop(to_drop, axis=1)


def nan_cols(df):
    # Sanity Check
    return {c for c, v in df.isna().any().to_dict().items() if v is True}


_ = """
################################################################################
############################# DATA PRE-PROCESSING ##############################
################################################################################

################################# MEDICINE HIST ################################
"""

temp = data['MedOrders'].copy()

"""
Keep:
– Medication
– Pharma class
– Therapeutic Class
– Ingredients

Add:
Y – "Duration of consumption"
N – maybe split "Ingredients" into bernoullis
N – Starting having medication X years ago
"""

intially_remove_these = ['Sig', 'Route', 'Disp', 'Unit', 'Refills',
                         'Frequency', 'Number of Times', 'Order Status',
                         'Order Class', 'Order Mode', 'Prescribing Provider',
                         'PatEncCsnCoded', 'Order Date', 'Order Age']
temp.drop(intially_remove_these, axis=1, inplace=True)

# Engineer duration of consumption
temp['End Date'] = pd.to_datetime(temp['End Date'])
temp['Start Date'] = pd.to_datetime(temp['Start Date'])
temp['Med_Duration_Days'] = (temp['End Date'] - temp['Start Date']).dt.days

remove_finally = ['Start Date', 'End Date', 'End Age']
temp.drop(remove_finally, axis=1, inplace=True)

# Medication pre-processing (maybe actually just keep one of the classes)
#   temp['Therapeutic Class'].isna().sum() / len(temp['Therapeutic Class'])
#   temp['Pharmaceutical Class'].isna().sum() / len(temp['Pharmaceutical Class'])
# There are many fewer therapeutic classes than Pharmaceutical classes. So, we're going
#   to keep the therapeutic classes.
temp['Medication'] = temp['Medication'].str.upper()
temp['Therapeutic Class'] = temp['Therapeutic Class'].str.upper()

drop_experimental = ['Medication', 'Ingredients', 'Pharmaceutical Class']
temp.drop(drop_experimental, axis=1, inplace=True)
temp.dropna(inplace=True)
temp = temp[temp['Med_Duration_Days'] >= 0].reset_index(drop=True)

# Per patient, generate count of each therapeutic class
temp['count'] = 1
avg_days_per_class = temp.groupby(['Patient Id', 'Therapeutic Class']).agg('mean')
avg_days_per_class = avg_days_per_class.pivot_table('Med_Duration_Days', ['Patient Id'], 'Therapeutic Class')
avg_days_per_class.columns = 'AvgDaysOn_' + avg_days_per_class.columns
avg_days_per_class = avg_days_per_class.dropna(axis=1, how='all').fillna(0.)
avg_days_per_class.reset_index(inplace=True)

# NOTE: THIS DATASET IS PERFECT
data['MedOrders'] = avg_days_per_class.copy()
# nan_cols(data['MedOrders'])

_ = """
################################# DEMOGRAPHICS #################################
"""

to_remove_demographics = ['Disposition',
                          'Marital Status',
                          'Interpreter Needed',
                          'Insurance Name',
                          'Insurance Type',
                          'Zipcode',
                          'Death Date SSA Do Not Disclose',
                          'Comment',
                          'City',
                          'Tags',
                          'Smoking History Taken',
                          'Date of Death']
# 'Smoking Hx']
data['Demographics'].drop(to_remove_demographics, axis=1, inplace=True)
data['Demographics']['Date of Birth'] = pd.to_datetime(data['Demographics']['Date of Birth'])
data['Demographics']['Current_Age'] = (datetime.now() - data['Demographics']['Date of Birth']).dt.days / 365.25
# Encoding some demographics
data['Demographics']['Gender'] = cat_to_num(data['Demographics']['Gender'])
data['Demographics']['Deceased'] = cat_to_num(data['Demographics']['Deceased'])
data['Demographics']['Race'] = cat_to_num(data['Demographics']['Race'])
data['Demographics']['Ethnicity'] = cat_to_num(data['Demographics']['Ethnicity'])
# Manual dtype correction
data['Demographics']['Smoking Hx'] = data['Demographics']['Smoking Hx'].str[0].astype(np.number)
# Sanitation: correct current ages of people that are deceased
data['Demographics']['Current_Age'] = data['Demographics'].apply(lambda row: row['Current_Age']
                                                                 if row['Deceased'] == 0 else row['Age at Death'],
                                                                 axis=1)

# NOTE: THIS IS A PERFECT DATASET
data['Demographics'] = data['Demographics'].select_dtypes(include=np.number)
# nan_cols(data['Demographics'])
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
data['Demographics'].drop(['Recent Height cm', 'Recent Weight kg'], axis=1, inplace=True)
data['Demographics'].drop(['Notes'], axis=1, inplace=True)
data['Demographics']['Smoking Hx'] = data['Demographics']['Smoking Hx'].fillna(
    data['Demographics']['Smoking Hx'].mean())

data['Demographics']['Recent BMI'] = data['Demographics']['Recent BMI'].fillna(
    data['Demographics']['Recent BMI'].mean())

_ = """
################################### DIAGNOSES ##################################
"""

to_remove_diagnoses = ['Source',
                       'ICD9 Code',
                       'Billing Provider',
                       'PatEncCsnCoded',
                       'Type',
                       'Description',
                       'Performing Provider']
data['Diagnoses'].drop(to_remove_diagnoses, axis=1, inplace=True)
# The age associated with a diagnosis is not necessarily the same as the current age of patient
data['Diagnoses']['Age_Of_Diagnosis'] = data['Diagnoses']['Age']

"""NOTE: Important; engineering TERM for our prediction_column,
         time until complication (if at all) here"""
# T86.11 is the code for kidney transplant rejection. Only get those specific diagnoses.
ppl_final_rejection = data['Diagnoses'][data['Diagnoses']['ICD10 Code'] == 'T86.11'].reset_index(drop=True)
ppl_final_rejection.drop_duplicates(subset=['Patient Id'], keep='first', inplace=True, ignore_index=True)
TIME_COMPLICATION_MAPPING = ppl_final_rejection.set_index('Patient Id')['Age_Of_Diagnosis'].to_dict()
# NOTE: `VAL_FOR_NO_COMPLICATION_YET` somewhat arbitrary, just ensured nobody had this value
VAL_FOR_NO_COMPLICATION_YET = 100

""" SEGMENT 2 """
temp = data['Diagnoses'].copy()

# Per patient, generate count of each ICD10 Code
temp['count'] = 1
temp['ICD10 Code'] = temp['ICD10 Code'].astype(str)
# Remove all the Z (Factors influencing health status and contact with health services) codes
temp = temp[~temp['ICD10 Code'].str.startswith('Z')]
track_code_freq = temp.groupby(['Patient Id', 'ICD10 Code']).agg({'count': 'count'})
# 'Age_Of_Diagnosis': 'mean'})
track_code_freq = track_code_freq.pivot_table('count', ['Patient Id'], 'ICD10 Code')

# Remove columns that don't matter too much
# The 0.85 can be a hyper-parameter
num_nas = {}
for col in track_code_freq.columns:
    num_nas[col] = track_code_freq[col].isna().sum() / len(track_code_freq[col])
num_nas = {col: value for col, value in num_nas.items() if value <= 0.85}
# np.quantile(list(num_nas.values()), 0.5)
# plt.hist(list(num_nas.values()), bins=20)
temp = track_code_freq[list(num_nas.keys())].fillna(0.)
temp.columns = 'CountOf_' + temp.columns
temp.reset_index(inplace=True)

""" SEGMENT 3 """
# Now add the column we engineered:
temp['Age_of_Complication'] = temp['Patient Id'].apply(
    lambda x: TIME_COMPLICATION_MAPPING.get(x, VAL_FOR_NO_COMPLICATION_YET))
temp['Had_Complication'] = temp['Age_of_Complication'].apply(lambda x: not (x == VAL_FOR_NO_COMPLICATION_YET))
# temp['Age_of_Complication'].hist()

# NOTE: This dataset is perfect
data['Diagnoses'] = temp.copy()
# Don't expect anything anyways in this case, all the NaNs were set to 0
# nan_cols(data['Diagnoses'])

_ = """
##################################### LABS #####################################
"""

to_remove_labs = ['Order Date',
                  'Result Date',
                  'Authorizing Provider',
                  'PatEncCsnCoded']
data['Labs'].drop(to_remove_labs, axis=1, inplace=True)
temp = data['Labs'].copy()
temp = temp.dropna(subset=['Abnormal'])
temp['Age_Result'] = temp['Age']
temp = temp[['Patient Id', 'Lab', 'Abnormal', 'Age_Result']]
temp['result_count'] = 1
lab_aggregations = temp.groupby(['Patient Id', 'Lab', 'Abnormal']).agg({'Age_Result': np.mean,
                                                                        'result_count': np.sum})
# avg_days_per_class = avg_days_per_class.pivot_table('Med_Duration_Days', ['Patient Id'], 'Therapeutic Class')

# Breaks the multi-index hierarchy
encoded_table = lab_aggregations.pivot_table('result_count', ['Patient Id'], ['Lab', 'Abnormal']).fillna(0)
encoded_table.columns = encoded_table.columns.map(' >> '.join).str.strip(' >> ')

results = {col: (encoded_table[col] == 0.0).sum() for col in encoded_table.columns}
# 1150 (can be a hyper-parameter) is a somewhat arbitrary, but based on histogram
results = {col: value for col, value in results.items() if value <= 1150}
most_common_columns = list(results.keys())
encoded_table = encoded_table[most_common_columns]
# VALUE IN THE TABLE REPRESENT NUMBER OF OCCURENCES

# NOTE: THIS IS A PERFECT DATASET
data['Labs'] = encoded_table.copy()
# Don't expect anything anyways in this case, all the NaNs were set to 0
# nan_cols(data['Labs'])

_ = """
################################## PROCEDURES ##################################
"""
""" SECTION 1 """
temp = data['Procedures'].copy()
TRANSPLANT_CODES = ['0TY10Z0', '0TY00Z0', '55.69']
# Only get transplant procedures
transplant_procedures = temp[temp['Code'].isin(TRANSPLANT_CODES)].drop_duplicates(
    subset=['Patient Id'], keep='first').reset_index(drop=True)
AGE_OF_TRANSPLANT_MAPPING = transplant_procedures.set_index('Patient Id')['Age'].to_dict()
temp = temp[['Patient Id', 'Age', 'Code', 'Description']]

""" SECTION 2 """
temp[temp['Description'].str.contains('transplant') & temp['Description'].str.contains('kidney')]['Code'].unique()
temp['count'] = 1
track_code_freq = temp.groupby(['Patient Id', 'Code']).agg({'count': 'count'})
# 'Age_Of_Diagnosis': 'mean'})
track_code_freq = track_code_freq.pivot_table('count', ['Patient Id'], 'Code')

# TODO: Need to find time of transplant

""" SECTION 3 """
num_nas = {}
for col in track_code_freq.columns:
    num_nas[col] = track_code_freq[col].isna().sum() / len(track_code_freq[col])
num_nas = {col: value for col, value in num_nas.items() if value <= 0.85}
# np.quantile(list(num_nas.values()), 0.5)
# plt.hist(list(num_nas.values()), bins=20)
temp = track_code_freq[list(num_nas.keys())].fillna(0.)
temp.columns = 'CountOf_' + temp.columns
temp.reset_index(inplace=True)
temp['Age_of_Transplant'] = temp['Patient Id'].apply(lambda x: AGE_OF_TRANSPLANT_MAPPING.get(x, np.nan))
temp = temp.dropna(subset=['Age_of_Transplant'], how='any')

data['Procedures'] = temp.copy()
# nan_cols(data['Procedures'])

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

# # Final sanity Check
# nan_cols(data['MedOrders'])
# nan_cols(data['Demographics'])
# nan_cols(data['Diagnoses'])
# nan_cols(data['Labs'])
# nan_cols(data['Procedures'])

_ = """
################################################################################
################################### MERGE ##################################
################################################################################
"""

"""MERGE MEDICAL HISTORY + DEMOGRAPHICS"""
merged_one = data['MedOrders'].join(data['Demographics'], how='inner', on=[
                                    'Patient Id'], lsuffix='_DELETE', rsuffix='_demo')
merged_one.drop('Patient Id_DELETE', axis=1, inplace=True)

""" MERGED FIRST_MERGED + DIAGNOSES """
merged_two = merged_one.join(data['Diagnoses'], how='inner', on=['Patient Id'],
                             lsuffix='_DELETE', rsuffix='_diagnosis')
merged_two.drop('Patient Id_DELETE', axis=1, inplace=True)

""" MERGED SECOND_MERGED + LABS """
merged_three = merged_two.join(data['Labs'], how='inner', on=['Patient Id'],
                               lsuffix='_DELETE', rsuffix='_labs')
# merged_three.drop('Patient Id_DELETE', axis=1, inplace=True)

""" MERGED THIRD_MERGED + PROCEDURES """  # TODO
merged_four = merged_three.join(data['Procedures'], how='inner', on=['Patient Id'],
                                lsuffix='_DELETE', rsuffix='_procedures')

_ = """
################################################################################
########################## FINAL, POST-MERGE DELETIONS #########################
################################################################################
"""
merged_four.drop(['Patient Id_demo', 'CountOf_nan', 'Patient Id_procedures',
                  'Patient Id_diagnosis', 'Patient Id_DELETE'], axis=1, inplace=True)


def assign_age_complication(row):
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


merged_four['Age_of_Complication'] = merged_four.apply(assign_age_complication, axis=1)
merged_four['Current_Age']
merged_four['Time_Until_Complication'] = merged_four['Age_of_Complication'] - merged_four['Age_of_Transplant']
merged_four = merged_four[merged_four['Time_Until_Complication'] >= 0].reset_index(drop=True)

_ = """
################################################################################
################################### SAVE FILE ##################################
################################################################################
"""
# Final pass to collectively analyse all columns and remove high correlations
FINAL_UNCORR = remove_highly_correlated(merged_four, skip=KEEP_NO_MATTER_WHAT, THRESHOLD=0.7)
# REORDER COLUMNS SO `Time_Until_Complication` IS AT THE VERY END
new_order = [c for c in list(FINAL_UNCORR) if c not in ['Patient Id',
                                                        'Time_Until_Complication']] + ['Time_Until_Complication']
FINAL_UNCORR = FINAL_UNCORR[new_order]
FINAL_COPY = FINAL_UNCORR.copy()

""" LOGIC FOR SUPERVISED, IMPUTED VALUES """
# FINAL_UNCORR = pd.read_csv('Data/Merged Complete/Core_Dataset.csv').infer_objects().drop('Unnamed: 0', axis=1)
FINAL_UNCORR.drop(['Age_of_Complication', 'Age_of_Transplant',
                  'Age at Death', 'Had_Complication'], axis=1, inplace=True)

# Save
FINAL_UNCORR.to_pickle(SAVE_PATH_FULL + '_SUPERVISED' + '.pkl')
FINAL_UNCORR.to_csv(SAVE_PATH_FULL + '_SUPERVISED' + '.csv')

""" LOGIC FOR SEMI-SUPERVISED, IMPUTED VALUES """
# Can't depend on Had_Complication for final None assignment (just look for `VAL_FOR_NO_COMPLICATION_YET` value)
FINAL_COPY['Time_Until_Complication'] = FINAL_COPY['Age_of_Complication'].apply(
    lambda x: np.nan if x == VAL_FOR_NO_COMPLICATION_YET else x)
FINAL_COPY.drop(['Age_of_Complication', 'Age_of_Transplant',
                 'Age at Death', 'Had_Complication'], axis=1, inplace=True)

FINAL_COPY.to_pickle(SAVE_PATH_FULL + '_SEMI-SUPERVISED' + '.pkl')
FINAL_COPY.to_csv(SAVE_PATH_FULL + '_SEMI-SUPERVISED' + '.csv')

# Rhea stuff
# FINAL_COPY['Time_Until_Complication'].hist()
# temp = FINAL_COPY['Time_Until_Complication'].dropna()
# np.quantile(temp, 0.40)

# Number of Unlabeled instances 0.68 = FINAL_UNCORR['Time_Until_Complication'].isna().sum() / len(FINAL_UNCORR['Time_Until_Complication'])
# EOF
