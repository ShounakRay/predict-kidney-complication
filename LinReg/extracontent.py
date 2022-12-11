# @Author: shounak
# @Date:   2022-11-23T17:26:32-08:00
# @Email:  shounak@stanford.edu
# @Filename: extracontent.py
# @Last modified by:   shounak
# @Last modified time: 2022-12-10T14:23:36-08:00


def to_hours(series):
    return series.dt.total_seconds() / (60 * 60.)


def to_days(series):
    return series.dt.total_seconds() / (60 * 60 * 24.)


def to_years(series):
    return series.dt.total_seconds() / (60 * 60 * 24. * 365.25)


def get_single_code(code):
    try:
        req = requests.get(f'https://www.aapc.com/codes/cpt-codes/{code}', HEADERS)
        soup = BeautifulSoup(req.content, 'html.parser')
        return soup.find('h1').text.strip().split(',  ')[-1]
    except Exception:
        pass


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

""" GETTING TOP FEATURES FOR THE FIRST TIME """
# cols = [c for c in ALL_DATA.columns if c in BETTER_DATA.columns]
# df_cols = pd.read_csv(
#     'Data/Merged Complete/Core_Dataset_SUPERVISED.csv').infer_objects().drop('Unnamed: 0', axis=1).columns
# mapping = dict(zip([i for i in range(len(df_cols))], df_cols))
# to_keep = [mapping.get(col) for col in cols if mapping.get(col) is not None] + ['Time_Until_Complication']
#
# pd.read_csv('Data/Merged Complete/Core_Dataset_SEMI-SUPERVISED.csv')[to_keep].to_csv(
#     'Data/Merged Complete/Core_Dataset_REDUCED-SEMI-SUPERVISED.csv')

# model = LinearRegression(StandardScaler(), ...)

""" BREAK: GET TOP FEATURE NAMES """
# df = pd.read_csv('Data/Merged Complete/Core_Dataset_REDUCED-SEMI-SUPERVISED.csv').infer_objects()
# df.drop('Unnamed: 0', axis=1, inplace=True)
# # (df['Time_Until_Complication'].dropna() / 12).describe()
# dict(zip(list(range(len(list(df)))), list(df)))
#
# list(pd.read_csv('Data/Merged Complete/Core_Dataset_SEMI-SUPERVISED.csv'))
#
# diagnoses_codes = [c.replace('CountOf_', '') for c in list(df) if c.startswith('CountOf_')
#                    and c.replace('CountOf_', '')[0].isalpha()]
# diagnoses_desc = [icd10.find(code).description for code in diagnoses_codes]
# dict(zip(diagnoses_codes, diagnoses_desc))
# diagnoses_blockdesc = [icd10.find(code).block_description for code in diagnoses_codes]
# Counter(diagnoses_blockdesc)
#
# HEADERS = {
#     'Access-Control-Allow-Origin': '*',
#     'Access-Control-Allow-Methods': 'GET',
#     'Access-Control-Allow-Headers': 'Content-Type',
#     'Access-Control-Max-Age': '3600',
#     'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
# }
# procedure_codes = [c.replace('CountOf_', '') for c in list(df) if c.startswith('CountOf_')
#                    and not c.replace('CountOf_', '')[0].isalpha()]
# procedure_descriptions = [get_single_code(code) for code in procedure_codes]
# dict(zip(procedure_codes, procedure_descriptions))
# Counter(procedure_descriptions)
#
""" CAITLIN: NEURAL NETWORK WEIGHT VISUALIZATION """
# d_MEDIAN = {24: 0.22037294531067073, 0: 0.21255028521797525, 19: 0.20010609027623966,
#             26: 0.1896462452147411, 21: 0.18898196128591727, 28: 0.18303140164314224,
#             4: 0.17827730369122785, 14: 0.174734026494632, 20: 0.17373482501320614,
#             7: 0.17113311654367158, 18: 0.16945354568694304, 12: 0.16193712679249808,
#             15: 0.1616828329850945, 16: 0.1594530242132901, 9: 0.1571537626423751,
#             25: 0.14190821664373993, 13: 0.13956102930702485, 3: 0.1369905336912253,
#             17: 0.1359449644341171, 8: 0.13139718081253746, 6: 0.12954042002946495,
#             1: 0.12765592844144397, 2: 0.12418838156901799,
#             2: 0.11984659298813953, 11: 0.11636217442632063, 10: 0.10770258828196504,
#             23: 0.10594526909281676, 5: 0.09997356445903631, 27: 0.07561061543683424}
# _ = plt.hist(d_MEDIAN.values(), alpha=0.6, color='green')
# d_MEAN = {24: 0.22037294531067073, 0: 0.21255028521797525, 19: 0.20010609027623966,
#           26: 0.1896462452147411, 21: 0.18898196128591727, 28: 0.18303140164314224,
#           4: 0.17827730369122785, 14: 0.174734026494632, 20: 0.17373482501320614,
#           7: 0.17113311654367158, 18: 0.16945354568694304, 12: 0.16193712679249808,
#           15: 0.1616828329850945, 16: 0.1594530242132901, 9: 0.1571537626423751,
#           25: 0.14190821664373993, 13: 0.13956102930702485, 3: 0.1369905336912253,
#           17: 0.1359449644341171, 8: 0.13139718081253746, 6: 0.12954042002946495,
#           1: 0.12765592844144397, 2: 0.12418838156901799, 22: 0.11984659298813953,
#           11: 0.11636217442632063, 10: 0.10770258828196504, 23: 0.10594526909281676,
#           5: 0.09997356445903631, 27: 0.07561061543683424}
# _ = plt.hist(d_MEAN.values(), alpha=0.6, color='red')
# top_percentile = np.percentile(list(d_MEDIAN.values()), 75)
# [mapping.get(ft) for ft, imp in d_MEDIAN.items() if imp >= top_percentile]
