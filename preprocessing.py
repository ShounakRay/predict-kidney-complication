# @Author: shounak
# @Date:   2022-10-28T11:07:35-07:00
# @Email:  shounak@stanford.edu
# @Filename: preprocessing.py
# @Last modified by:   shounak
# @Last modified time: 2022-10-28T11:21:10-07:00

import pandas as pd

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

# EOF
