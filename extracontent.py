# @Author: shounak
# @Date:   2022-11-23T17:26:32-08:00
# @Email:  shounak@stanford.edu
# @Filename: extracontent.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-26T12:48:08-08:00


def to_hours(series):
    return series.dt.total_seconds() / (60 * 60.)


def to_days(series):
    return series.dt.total_seconds() / (60 * 60 * 24.)


def to_years(series):
    return series.dt.total_seconds() / (60 * 60 * 24. * 365.25)


""" MODEL EDA """
# # Plot Histogram for Understanding
# axarr = ALL_DATA.hist(bins=20, figsize=(100, 100))
# for ax in axarr.flatten():
#     ax.set_xlabel("")
#     ax.set_ylabel("")
# plt.savefig('Images/Data_Histogram.png')

""" NORMALIZATION """
# def NormalizeData(data):
#     # https://stackoverflow.com/questions/18380419/normalization-to-bring-in-the-range-of-0-1
#     return (data - np.min(data)) / (np.max(data) - np.min(data))
# # Normalize Errors
# normalized_train_errs = NormalizeData(list(training_std_err.values()))
# normalized_test_errs = NormalizeData(list(testing_std_err.values()))


""" RHEA THIS BIT IS FOR YOU """
# Rhea stuff
# FINAL_COPY['Time_Until_Complication'].hist()
# temp = FINAL_COPY['Time_Until_Complication'].dropna()
# np.quantile(temp, 0.40)

""" OLD TRANSPLANT STUFF """
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

""" OLD DIAGNOSES STUFF """
# only_rejected.drop(['ICD10 Code', 'Age'], axis=1, inplace=True)
# """ONLY GET THE LAST VISIT"""
# # Only gets the last visit (according to "Date" column)
# last_visit = data['Diagnoses'].sort_values(['Patient Id', 'Date']).groupby(
#     'Patient Id').apply(lambda x: x.iloc[[-1]]).reset_index(drop=True)
