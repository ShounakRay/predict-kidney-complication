# @Author: shounak
# @Date:   2022-10-28T11:07:35-07:00
# @Email:  shounak@stanford.edu
# @Filename: preprocessing.py
# @Last modified by:   shounak
# @Last modified time: 2022-10-28T20:26:59-07:00

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FPATHS = {'Notes': 'Data/kds1_clinical_note-002.csv',
          'Demographics': 'Data/kds1_demographics.csv',
          'Diagnoses': 'Data/kds1_diagnoses.csv',
          'Labs': 'Data/kds1_labs-001.csv',
          'MedOrders': 'Data/kds1_med_orders.csv',
          'Codebook': 'Data/kds1_patientCodebook.csv',
          'Procedures': 'Data/kds1_procedures.csv'}

data = {}
for label, path in FPATHS.items():
    data[label] = pd.read_csv(path).infer_objects()
    print(f'Ingested "{label}" dataset.')
COPY_DATA = data.copy()

# Ignoring NOTES for now

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

REDUCED_DEMOGRAPHICS = data['Demographics'].select_dtypes(include=np.number)

to_remove_diagnoses = ['Source',
                       'ICD9 Code',
                       'Billing Provider',
                       'PatEncCsnId']
data['Diagnoses'].drop(to_remove_diagnoses, axis=1, inplace=True)


only_rejection = data['Diagnoses'][data['Diagnoses']['ICD10 Code'] == 'T86.11'].reset_index(drop=True)
last_visit = only_rejection.sort_values(['Patient Id', 'Date']).groupby(
    'Patient Id').apply(lambda x: x.iloc[[-1]]).reset_index(drop=True)
last_visit.drop(['Type', 'ICD10 Code', 'Description', 'Performing Provider'], axis=1, inplace=True)

only_rejection.select_dtypes(include=np.number)


to_remove_labs = ['Order Date',
                  'Result Date',
                  'Authorizing Provider',
                  'PatEncCsnId']
data['Labs'].drop(to_remove_labs, axis=1, inplace=True)

# Get rid of rows that definitely don't matter
# Engineer column for time until complication (if at all) since transplant procedure
# Corroborate description and ICD10 codes

transplant_people = data['Procedures'][data['Procedures']['Code'].isin(['0TY10Z0', '0TY00Z0'])].reset_index(drop=True)

merged = transplant_people.join(last_visit, how='inner', on=['Patient Id'],
                                lsuffix='_transplant', rsuffix='_complication')

merged.drop('Patient Id_transplant', axis=1, inplace=True)
merged['time_diff'] = merged['Age_transplant'] - merged['Age_complication']
merged = merged.infer_objects()
merged = merged.select_dtypes(include=np.number).drop(
    ['Performing Provider', 'Billing Provider', 'PatEncCsnId', 'Patient Id_complication'], axis=1)

JOINED_FINAL = REDUCED_DEMOGRAPHICS.join(merged, how='inner', on=['Patient Id'],
                                         lsuffix='_demographics',
                                         rsuffix='_common').drop(['Patient Id_common',
                                                                  'Patient Id_demographics'], axis=1)
JOINED_FINAL.drop(['Age at Death', 'Notes', 'Patient Id'], axis=1, inplace=True)

train_size = int(len(JOINED_FINAL) * 0.7)
reduced = JOINED_FINAL.sample(frac=1).reset_index(drop=True)
training_set = reduced.values[:train_size, :]
test_set = reduced.values[train_size:, :]

model = LinearRegression()
x = training_set[:, :-1]
y = training_set[:, -1]
model.fit(x, y)
model.score(x, y)

model.predict(test_set[:, :-1])
test_set[:, -1]
# EOF
