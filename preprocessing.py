# @Author: shounak
# @Date:   2022-10-28T11:07:35-07:00
# @Email:  shounak@stanford.edu
# @Filename: preprocessing.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-23T21:18:08-08:00

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

FPATHS = {
    # 'Notes': 'Data/kds1_clinical_note-002.csv',
    'Demographics': 'Data/kds1_demographics.csv',
    'Diagnoses': 'Data/kds1_diagnoses.csv',
    'Labs': 'Data/kds1_labs-001.csv',
    'MedOrders': 'Data/kds1_med_orders.csv',
    # 'Codebook': 'Data/kds1_patientCodebook.csv',
    'Procedures': 'Data/kds1_procedures.csv'}

data = {}
for label, path in FPATHS.items():
    data[label] = pd.read_csv(path).infer_objects()
    print(f'Ingested "{label}" dataset.')
COPY_DATA = data.copy()

# data['Codebook'] maps patient names to MRN code
# data['Notes'] has patient clininal meeting notes


def cat_to_num(series):
    return series.astype('category').cat.codes


_ = """
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
                         'PatEncCsnId', 'Order Date', 'Order Age']
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


_ = """
################################################################################
################################# DEMORGAPHICS #################################
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
                                                                 if row['Deceased'] == 0 else -1, axis=1)
# NOTE: THIS IS A PERFECT DATASET
data['Demographics'] = data['Demographics'].select_dtypes(include=np.number)

_ = """
################################################################################
################################### DIAGNOSES ##################################
"""

to_remove_diagnoses = ['Source',
                       'ICD9 Code',
                       'Billing Provider',
                       'PatEncCsnId',
                       'Type',
                       'Description',
                       'Performing Provider']
data['Diagnoses'].drop(to_remove_diagnoses, axis=1, inplace=True)
# The age associated with a diagnosis is not necessarily the same as the current age of patient
data['Diagnoses']['Age_Of_Diagnosis'] = data['Diagnoses']['Age']
# T86.11 is the code for kidney transplant rejection. Only get those specific diagnoses.
ppl_final_rejection = data['Diagnoses'][data['Diagnoses']['ICD10 Code'] == 'T86.11'].reset_index(drop=True)
ppl_that_got_rejected = ppl_final_rejection['Patient Id'].unique()
# data['Diagnoses']['Rejected'] = data['Diagnoses']['Patient Id'].isin(ppl_that_got_rejected)

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
plt.hist(list(num_nas.values()), bins=20)
temp = track_code_freq[list(num_nas.keys())].fillna(0.)
temp.columns = 'CountOf_' + temp.columns

# only_rejected.drop(['ICD10 Code', 'Age'], axis=1, inplace=True)
# """ONLY GET THE LAST VISIT"""
# # Only gets the last visit (according to "Date" column)
# last_visit = data['Diagnoses'].sort_values(['Patient Id', 'Date']).groupby(
#     'Patient Id').apply(lambda x: x.iloc[[-1]]).reset_index(drop=True)

# NOTE: This dataset is perfect
data['Diagnoses'] = temp.copy()

_ = """
################################################################################
##################################### LABS #####################################
"""

to_remove_labs = ['Order Date',
                  'Result Date',
                  'Authorizing Provider',
                  'PatEncCsnId']
data['Labs'].drop(to_remove_labs, axis=1, inplace=True)
temp = data['Labs'].copy()
temp = temp.dropna(subset=['Abnormal'])
temp['Age_Result'] = temp['Age']
temp = temp[['Patient Id', 'Lab', 'Abnormal', 'Age_Result']]
temp['result_count'] = 1
lab_aggregations = temp.groupby(['Patient Id', 'Lab', 'Abnormal']).agg({'Age_Result': np.mean,
                                                                        'result_count': np.sum})
avg_days_per_class = avg_days_per_class.pivot_table('Med_Duration_Days', ['Patient Id'], 'Therapeutic Class')

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

_ = """
################################################################################
################################## PROECURES ###################################
"""

# Only get PROCEDURES of people that got left and right kidneys transplanted
transplant_people = data['Procedures'][data['Procedures']
                                           ['Code'].isin(['0TY10Z0', '0TY00Z0'])].reset_index(drop=True)

_ = """
################################################################################
################################### MERGE ##################################
"""

"""MERGE MEDICAL HISTORY + DEMOGRAPHICS"""
merged_one = data['MedOrders'].join(data['Demographics'], how='inner', on=[
                                    'Patient Id'], lsuffix='_DELETE', rsuffix='demo_')
merged_one.drop('Patient Id_DELETE', axis=1, inplace=True)

""" MERGED FIRST_MERGED + DIAGNOSES """
merged_two = merged_one.join(data['Diagnoses'], how='inner', on=['Patient Id'])

""" MERGED SECOND_MERGED + LABS """
merged_three = merged_two.join(data['Labs'], how='inner', on=['Patient Id'])

""" MERGED THIRD_MERGED + PROCEDURES """  # TODO
# merged_four = merged_three.join(data['Procedures'], how='inner', on=['Patient Id'])


_ = """
################################################################################
################################### SAVE FILE ##################################
"""
# Pending

# EOF
