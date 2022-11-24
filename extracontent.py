# @Author: shounak
# @Date:   2022-11-23T17:26:32-08:00
# @Email:  shounak@stanford.edu
# @Filename: extracontent.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-24T01:37:39-08:00


def to_hours(series):
    return series.dt.total_seconds() / (60 * 60.)


def to_days(series):
    return series.dt.total_seconds() / (60 * 60 * 24.)


def to_years(series):
    return series.dt.total_seconds() / (60 * 60 * 24. * 365.25)


# merged = transplant_people.join(data['Labs'], how='inner', on=['Patient Id'],
#                                 lsuffix='_transplant', rsuffix='_complication')
#
# merged.drop('Patient Id_transplant', axis=1, inplace=True)
# merged['time_diff'] = merged['Age_transplant'] - merged['Age_complication']
# merged = merged.infer_objects()
# merged = merged.select_dtypes(include=np.number).drop(
#     ['Performing Provider', 'Billing Provider', 'PatEncCsnId', 'Patient Id_complication'], axis=1)
#
# merged.drop_duplicates(subset=['Patient Id'], inplace=True)
# JOINED_FINAL = REDUCED_DEMOGRAPHICS.join(merged, how='inner', on=['Patient Id'],
#                                          lsuffix='_demographics',
#                                          rsuffix='_common').drop(['Patient Id_common',
#                                                                   'Patient Id_demographics'], axis=1)
# JOINED_FINAL.drop(['Age at Death', 'Notes', 'Patient Id'], axis=1, inplace=True)


# only_rejected.drop(['ICD10 Code', 'Age'], axis=1, inplace=True)
# """ONLY GET THE LAST VISIT"""
# # Only gets the last visit (according to "Date" column)
# last_visit = data['Diagnoses'].sort_values(['Patient Id', 'Date']).groupby(
#     'Patient Id').apply(lambda x: x.iloc[[-1]]).reset_index(drop=True)
