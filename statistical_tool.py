"""
data processing and analysis with Python
==================================

statistical_tool.py provides comprehensive tools to clean, process,
and perform data science/statistical learning on a provided dataset

Made by: Duy Nguyen (github.com/mathladd)

"""

import statsmodels.api as sm
import statsmodels.tools.eval_measures as smte
import statsmodels.tools.sm_exceptions as sm_except

from sklearn.exceptions import ConvergenceWarning
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.linear_model import LinearRegression, ElasticNetCV, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mord import LogisticAT

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
import matplotlib.pyplot as plt

import os
import warnings
import gc

pd.set_option('display.expand_frame_repr', False)

plt.rcParams.update({'font.size': 8})
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=sm_except.ConvergenceWarning)


# ************************** GLOBAL SETTINGS: START HERE *****************************************
# ---------------- GENERAL SETTINGS ----------------
LINK_TO_DAT = r'linkToACleanDataset.csv'    # link to data you want to clean and do analysis on
FRAC = 1  # OPTIMIZATION: Get sample of a fraction of the data
ALL_FEATURE_TYPES = {

}  # OPTIMIZATION: Customize each feature's type in a dictionary (e.g. {'a': np.float64}) to reduce memory usage
EXCEL_SHEET_INDEX = 0  # For excel files only. Set index of sheet to be read

# ---------------- DATA PROCESSING SETTINGS ----------------
LIST_TO_DATETIME = [

]  # List of features to be converted to datetime
LIST_TO_STANDARDIZE = [

]  # List of features to be standardized (x = (x-mean)/sigma), or...
BOOLEAN_STANDARDIZE_ALL_NUMERIC = False  # Standardize all possible numeric features
LIST_OF_INTERACTIONS = [

]  # List of interactions to add e.g. ['Age:Fare', 'Age:Health'], or...
ALL_INTERACTIONS = False  # Add all possible interactions
DICT_OF_POLIES = {

}  # Dict of features and degrees to add e.g. {'Age':3} (y = a + Age + Age^2 + Age^3), or...
ALL_POLY_DEGREES = 1  # Power up all features to n degree
LIST_TO_DROP = [

]  # List of variables to be dropped e.g. ['Unnamed: 0', 'col20']

# ---------------- MODEL SETTINGS ----------------
REGRESSION = True  # Set True if you want to do regression analyses
CLASSIFICATION = False  # Set True if you want to do classification analyses
TARGET_NAME = ''  # The dependent/target name, if not set, will be prompted during modeling
ALPHA = 0.05  # For confidence interval
CV = 10  # Set to 1 to eliminate cross validation. Set to n to do LOOCV
DISPLAY_GRAPH = False  # Set to False to stop displaying graphs
INDIVIDUAL_FEATURE_ANALYSIS = True

# Regularization settings
PENALTY = 0  # Penalty for regularization. If set to 0, will be set automatically by sklearn
L1_RATIO = 0.5  # Set to 1 to perform purely lasso regularization, 0 for pure ridge, and inbetween for mix of both

# Quad test
QUAD_TEST_TO_DEGREE = 3  # Degree to quad test to (regression: lowest MSE; classification: highest accuracy)

# Classification settings
LIST_OF_CLASS_WEIGHTS = []
N_NEIGHBORS = 10  # Number of neighbors for K-nearest neighbors classifier


# ************************************************** MAIN *******************************************************
def main(link_to_dat, excel_sheet_index, frac, all_feature_types,
         list_to_standardize, boolean_standardize_all_numeric, list_to_datetime,            # data processing
         regression, list_to_drop, dict_of_poly, list_of_interactions, all_interactions,    # regression
         all_poly_degrees, cv, l1_ratio, penalty, quad_test_to_degree, display_graph,       # regression (cont.)
         classification, list_of_class_weights, individual_feature_analysis, n_neighbors):
    df, not_csv_alerts = df_import(link_to_dat=link_to_dat,
                                   excel_sheet_index=excel_sheet_index,
                                   frac=frac,
                                   all_feature_types=all_feature_types)
    if df.empty:
        print('Empty dataframe.')
        return
    print('\nFinished importing data.\n')

    target_name = TARGET_NAME
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_display_raw(df)
    df, dfs_to_export = df_process(df)
    x, y = df_analysis(df_input=df,
                       target_name=target_name,
                       link_to_dat=link_to_dat,
                       not_csv_alerts=not_csv_alerts,
                       dfs_to_export=dfs_to_export,
                       list_to_standardize=list_to_standardize,
                       boolean_standardize_all_numeric=boolean_standardize_all_numeric,
                       list_to_datetime=list_to_datetime)
    gc.collect()

    # --------------------------------------------- REGRESSION ------------------------------------------------------- #
    #  Questions to ask:                                                                                               #
    #   1. Is there a relationship between the target and any of the variables? (use F-stat in multiple linear reg)    #
    #   2. How strong is the relationship? (use adjusted R-squared in multiple linear reg)                             #
    #   3. Which variables should be included? (use each var's t-stat and p-value in multiple linear reg)              #
    #   4. How large does each variable affect the target? (use each var's confidence interval of coefficient in       #
    #                                                               multiple linear reg, as well as in individual reg) #
    #   5. How accurate can we predict future target values? (use prediction interval to predict individual response   #
    #               - wider to account for irreducible errors - or confidence interval to predict on average response) #
    #   6. Is the relationship linear? (use studentized residual spot trend & outliers)                                #
    #                                                                                                                  #
    # --------------------------------------------- REGRESSION ------------------------------------------------------- #
    if regression:
        print(make_title('REGRESSION'))
        regression_x = x.copy()
        error_check, transformation_done = transform_main(
            x=regression_x, list_to_drop=list_to_drop, dict_of_poly=dict_of_poly,
            list_of_interactions=list_of_interactions,
            all_interactions=all_interactions, all_poly_degrees=all_poly_degrees
        )

        if error_check:
            print('ERRORS DURING TRANSFORMATION OF X')
            return

        if transformation_done:
            print(transformation_done)

        regression_main(
            x=regression_x, y=y, cv=cv, l1_ratio=l1_ratio, penalty=penalty,
            individual_feature_analysis=True, quad_test_to_degree=quad_test_to_degree, display_graph=display_graph
        )
    del regression_x
    gc.collect()

    # ---------------------------------------------- CLASSIFICATION -------------------------------------------------- #
    if classification:
        print(f"\n{make_title('CLASSIFICATION')}")
        classification_x = x.copy()
        error_check, transformation_done = transform_main(
            x=classification_x, list_to_drop=list_to_drop, dict_of_poly=dict_of_poly,
            list_of_interactions=list_of_interactions,
            all_interactions=all_interactions, all_poly_degrees=all_poly_degrees
        )

        if error_check:
            print('ERRORS DURING TRANSFORMATION OF X')
            return

        if transformation_done:
            print(transformation_done)

        classification_main(
            x=classification_x, y=y, cv=CV,
            list_of_class_weights=list_of_class_weights, interpretation_with_sm=True,
            individual_feature_analysis=individual_feature_analysis,
            quad_test_to_degree=quad_test_to_degree, n_neighbors=n_neighbors,
            display_graph=display_graph)

    print('\nEnd of program.')


# ********************************************* FUNCTIONS ****************************************************
def df_process(df):
    """ Custom user-added data processing """
    # 1. DATA CLEANING:
    #       PIVOT/UNPIVOT,
    #       NaN/REDUNDANCIES,
    #       INCONSISTENCY,
    #       MERGE/CONCATENATE,
    #       FILTER,
    #       QUALITATIVE
    # ------------------------- custom processing here ------------------------- #
    #
    df = df.drop('Unnamed: 0', axis=1)
    #
    # ----------------------- End of custom processing ----------------------- #
    # 2. EVALUATION:
    #       CALCULATED COLUMNS,
    #       GROUP BY,
    #       MAKING AND PRINTING CUSTOM CALCULATIONS
    # ------------------------- custom calculations here ------------------------- #
    #

    #
    # ----------------------- End of custom calculations ----------------------- #
    dfs_to_export = []  # Add dfs to be exported here (e.g. df1, df2...)
    return df, dfs_to_export


def df_import(link_to_dat, excel_sheet_index, frac, all_feature_types):
    """ Imports data from provided link (also accomodates excel files) """

    print(f'\nReading: {link_to_dat}')
    file_unique_display = ''
    file_size = os.stat(link_to_dat).st_size
    if not all_feature_types:
        all_feature_types = None

    # Read csv or txt
    if link_to_dat.find('.csv') != -1 or link_to_dat.find('.txt') != -1:

        dask_df = dd.read_csv(link_to_dat, dtype=all_feature_types)
        # read multiple files: dask_dfs = dd.read_csv('data_path/2014-*.csv')
        # export dask_df to single csv: dask_df.to_csv('path/to/csv.csv', single_file=True)

        if file_size > 1000000000:
            print('File is too large. Please try df_import() to perform MapReduce with Dask.')
            print(f'\nNumber of variables: {len(dask_df.columns)}')
            print(f'Sample:\n{dask_df.head()}')
            print('\nCreating the Client...')
            client = Client()
            print('Client created. Map reducing initialized...')

            if frac < 1:
                dask_df = dask_df.sample(frac=frac)
                print(f'Returning {int(frac * 100)}% of data')

            # -------------------------------- MAP REDUCING STARTS --------------------------------------
            # dask_df = dask_df[['NETWORK', CHANNEL]]
            # dask_df = dask_df.drop(["NETWORK", "INTERNAL_SEGMENTATION", "BANK_OR_UNBANK"], axis=1)
            # dask_df = dask_df.groupby(['GEOGRAPHY', 'BANK_VISA']).agg({'UNIQUE_USER': 'sum'}).reset_index()
            # dask_df = dask_df.assign(Z = dask_df["X"] + dask_df["Y"]) --> Add calculated col
            # dask_df = dask_df[dask_df["X"] > 0]                       --> Choose only rows with value in col "X" > 0
            # dask_df = dask_df.sample(frac=0.005)                      --> Sampling the data for faster loading

            # -------------------------------- MAP REDUCING ENDS ----------------------------------------
            dask_df = client.persist(dask_df)  # Store df into cluster's memory (note: cluster must have enough RAM)
            print('Finished map reducing. Printing additional computations...')

            # ---------------------------- CUSTOM HIGH-LEVEL COMPUTATIONS -----------------------------
            # print(dask_df['NUMBERS_USER'].sum().compute())

            # -------------------------------- COMPUTATIONS ENDS ----------------------------------------
            print('[Ended high-level query]')
            df = dask_df.compute().reset_index(drop=True)  # Store df into single machine's RAM !!
            print('Closing Client...')
            client.close()
        else:
            df = dask_df.compute().reset_index(drop=True)

    # Read excel files
    elif (link_to_dat.find('.xlsx') or link_to_dat.find('.xls')) != -1:
        df = pd.read_excel(link_to_dat, sheet_name=excel_sheet_index, dtype=all_feature_types)
        sheet_names_list = pd.ExcelFile(link_to_dat).sheet_names
        file_unique_display = \
            f"{'%-100s' % f'|  Displaying sheet: {str(excel_sheet_index + 1)}/{len(sheet_names_list)}'}|\n" \
            f"{'%-100s' % f'|  List of sheets: {str(sheet_names_list)}'}|\n"

    # Read hdf
    elif link_to_dat.find('.hdf') != -1:
        df = pd.read_hdf(link_to_dat, key='test', dtype=all_feature_types)

    # Unknown file type
    else:
        print('Cannot determine type of file')
        df = pd.DataFrame

    return df, file_unique_display


def df_export(dfs_to_export, list_link_to_original_df_folder, name_of_original_df):
    """ Exports processed df files
    :param dfs_to_export: All dfs to be exported
    :param list_link_to_original_df_folder: Link to original folder holding master df as list of folder names
    :param name_of_original_df: Name of orignal df all dfs_to_export derived from

    :return: None. All dfs exported
    """
    if dfs_to_export:

        # Prompt for file type to export
        while True:
            export_input = input('Export reports to csv or xlsx? (csv/xlsx): ')
            if 'csv' in export_input or 'xlsx' in export_input:
                break
            else:
                print('Unknown export file type.')

        # Export dfs to n csv files (n=len(dfs_to_export))
        if export_input == 'csv':
            csv_index = 0
            while csv_index < len(dfs_to_export):
                current_csv_file_link = list_link_to_original_df_folder.copy()
                current_csv_file_name = f"{name_of_original_df} REPORT {str(csv_index)}.csv"
                current_csv_file_link.append(current_csv_file_name)
                export_link = '\\'.join(current_csv_file_link)
                dfs_to_export[csv_index].to_csv(export_link, index=False)
                print(f'Exported: {current_csv_file_name}')
                csv_index += 1

        # Export dfs to one single excel file with n sheets (n=len(dfs_to_export))
        elif export_input == 'xlsx':
            current_xlsx_file_link = list_link_to_original_df_folder.copy()
            current_xlsx_file_name = f'{name_of_original_df} SHEETS_OF_REPORTS.xlsx'
            current_xlsx_file_link.append(current_xlsx_file_name)
            export_link = '\\'.join(current_xlsx_file_link)
            dfs_to_export[0].to_excel(excel_writer=export_link, sheet_name='REPORT 0', index=False)
            export_sheet_index = 1
            while export_sheet_index < len(dfs_to_export):
                with pd.ExcelWriter(export_link, mode='a') as writer:
                    dfs_to_export[export_sheet_index] \
                        .to_excel(writer, sheet_name=f'REPORT {str(export_sheet_index)}', index=False)
                export_sheet_index += 1
            print(f'Exported: {current_xlsx_file_name}')


def df_display_raw(df):
    """ Display df pre-processing """
    print('\n[RAW DATA FRAME DISPLAY])')
    print(df)
    print('[END OF RAW]')


def df_analysis(df_input, target_name, link_to_dat, not_csv_alerts, dfs_to_export, list_to_standardize,
                boolean_standardize_all_numeric, list_to_datetime):
    """ data analysis (regression and classification) """
    df = df_input.copy()
    all_feature_names = tuple(df.columns)
    number_of_features = len(all_feature_names)
    past_number_of_observations = len(df)
    all_dtypes = df.dtypes.astype(str).values
    unique_display_width = 150
    unique_display_half = int(unique_display_width / 2)

    features_uniques_report, features_characteristics_report = [], []
    categorical_features, object_features, numerical_features, constant_features, nan_features = [], [], [], [], []

    # Looping through each variable
    current_feature_index = 0
    while current_feature_index < number_of_features:
        current_feature_name = str(all_feature_names[current_feature_index])
        all_uniques = df[current_feature_name].unique()
        num_of_uniques = len(all_uniques)

        flag_problem, flag_int, flag_float, flag_date, flag_no_time = '', False, False, False, False

        # Sorting out data types and looking for object variables
        current_feature_dtype = all_dtypes[current_feature_index]
        if current_feature_dtype in ('int64', 'int32', 'int16', 'int8'):
            flag_int = True
            numerical_features.append(current_feature_name)
        elif current_feature_dtype in ('float64', 'float32', 'float16'):
            flag_float = True
            numerical_features.append(current_feature_name)
        elif current_feature_dtype == 'datetime64[ns]':
            current_feature_dtype = 'dt64'
            flag_date = True
        elif current_feature_dtype == 'object':
            flag_date_fail = False
            if current_feature_name in list_to_datetime:
                try:
                    df[current_feature_name] = df[current_feature_name].astype('datetime64[ns]')
                except ValueError:
                    flag_date_fail = True
            else:
                flag_date_fail = True
            if num_of_uniques < int(past_number_of_observations / 2) and flag_date_fail:
                df[current_feature_name] = df[current_feature_name].astype('category')
                all_dtypes[current_feature_index] = 'category'
                current_feature_dtype = 'cat'
                categorical_features.append(current_feature_name)
            else:
                object_features.append(current_feature_name)
                flag_problem = 'x'
        elif str(current_feature_dtype) == 'category':
            current_feature_dtype = 'cat'
            categorical_features.append(current_feature_name)

        # Adding constant column to be removed
        if num_of_uniques <= 2 and pd.isnull(all_uniques).any():
            constant_features.append(current_feature_name)

        # Looking for variables containing NaN
        percent_nan = ''
        if df[current_feature_name].isnull().any():
            percent_nan = str(round(((df[current_feature_name].isna().sum()) / past_number_of_observations) * 100, 1))
            nan_features.append(current_feature_name)
            flag_problem = 'x'

        # Basic statistics + distribution for quantitative, and the mode and frequency percent for qualitative data
        distribution = ''
        sum_num = ''

        # CHECK FOR INTEGER DATA TYPE
        if flag_float or flag_int:
            # Sum and descriptions
            sum_num = df[current_feature_name].sum()
            describe = df[current_feature_name].describe()
            describe_mode = df[current_feature_name].mode()[0]
            describe_mode_freq = df.loc[
                df[current_feature_name] == describe_mode, current_feature_name
            ].count()
            describe_mode_freq_percent = round((describe_mode_freq / past_number_of_observations) * 100, 1)

            # Converting integer descriptions
            if flag_int:
                for index in range(3, 8):
                    describe_to_int = int(describe[index])
                    describe[index] = describe_to_int

            # Summarize description
            description = (describe_mode, describe_mode_freq_percent, describe[1], describe[2],
                           describe[3], describe[4], describe[5], describe[6], describe[7])

            # Distribution of int and float data types
            if describe[7] != describe[3]:
                distribution_list = ['-'] * 24
                distribution_list.insert(
                    int(round(((describe[5] - describe[3]) / (describe[7] - describe[3])) * 24, 0)), '|')
                distribution_list.insert(
                    int(round(((describe[4] - describe[3]) / (describe[7] - describe[3])) * 24, 0)), '(')
                distribution_list.insert(
                    int(round(((describe[6] - describe[3]) / (describe[7] - describe[3])) * 24, 0)) + 2, ')')
                distribution = ''.join(distribution_list)
            else:
                distribution = '-' * (24 + 3)

        elif flag_date:
            describe = df[current_feature_name].describe()
            describe_mode = describe[2]
            describe_mode_freq_percent = round(describe[3] / past_number_of_observations * 100, 1)
            min_date = describe[4]
            max_date = describe[5]
            if len(df[current_feature_name][pd.notnull(df[current_feature_name])].dt.hour.unique()) == 1:
                flag_no_time = True
            if flag_no_time:
                describe_mode = str(describe_mode.date())
                min_date = str(min_date.date())
                max_date = str(max_date.date())
            description = (describe_mode, describe_mode_freq_percent, '', '', min_date, '', '', '', max_date)

        # object or categorical types
        else:
            describe = df[current_feature_name].describe()
            describe_mode = str(describe[2])
            describe_mode_freq_percent = round(describe[3] / past_number_of_observations * 100, 1)
            description = (describe_mode, describe_mode_freq_percent, '', '', '', '', '', '', '')

        # Unique values for each variable
        try:
            if percent_nan:
                all_uniques = tuple(np.append('nan', np.sort(all_uniques[pd.notnull(all_uniques)])))
            else:
                all_uniques = tuple(np.sort(all_uniques))
            uniques_sorted = 1
        except TypeError:
            if percent_nan:
                all_uniques = tuple(np.append('nan', np.sort(all_uniques[pd.notnull(all_uniques)].astype(str))))
            else:
                all_uniques = tuple(np.sort(all_uniques.astype(str)))
            uniques_sorted = 11
        all_uniques_as_string = str(all_uniques)

        # Creating unique show
        if len(all_uniques_as_string) < unique_display_width:
            unique_show = f'{all_uniques_as_string[:unique_display_half - 5]}' \
                          f'{all_uniques_as_string[unique_display_half - 5:]}'
        else:
            unique_show = f'{all_uniques_as_string[:unique_display_half - 5]} ... ' \
                          f'{all_uniques_as_string[-unique_display_half + 5:]}'

        # Summarizing each variable
        current_feature_unique_report = (flag_problem, current_feature_name, current_feature_dtype,
                                         num_of_uniques, uniques_sorted,
                                         ('%-' + str(unique_display_width - 5) + 's') % unique_show)
        current_feature_characteristics_report = (flag_problem, current_feature_name, percent_nan,
                                                  current_feature_dtype, num_of_uniques,
                                                  description[0], float(description[1]), sum_num,
                                                  description[2], description[3], description[4],
                                                  description[5], description[6], description[7], description[8],
                                                  distribution)

        # Adding summaries to reports
        features_uniques_report.append(current_feature_unique_report)
        features_characteristics_report.append(current_feature_characteristics_report)
        current_feature_index += 1

    features_uniques_report = pd.DataFrame(data=features_uniques_report,
                                           columns=('!', 'ID', 'type', 'uniques', 'sorted', 'uniques_show'))
    features_characteristics_report = pd.DataFrame(
        data=features_characteristics_report,
        columns=('!', 'ID', '%_NA', 'type', 'uniques', 'Mod', 'Md_F',
                 'Sum', 'Mean', 'std', 'Min', '0.25', 'Med', '0.75', 'Max', 'Distribution'))

    # REPORT
    print('\n[DATA FRAME POST-CLEANING]')
    print(df)
    print('[END OF DATA FRAME POST-CLEANING DISPLAY]')
    print(f"\n\nFEATURES SUMMARY: {past_number_of_observations} observations and {number_of_features} features")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           'display.max_colwidth', unique_display_width):
        print("[FEATURES' UNIQUES]")
        print(features_uniques_report)
        print('-' * 200)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("[FEATURES' CHARACTERISTICS]")
        print(features_characteristics_report)
        print(f'\nThere are {len(object_features)} object features: {object_features}')
        print(f'There are {len(nan_features)} features with NaN values: {nan_features}')
        print(f'All dtypes: {dict(zip(all_feature_names, all_dtypes))}')
        print('\n' + '\t' * 12 + '---End---')

    link_to_folder = link_to_dat.split('\\')
    name_of_current_file = link_to_folder.pop(-1)
    name_of_current_file_splitted = name_of_current_file.split('.')
    name_of_current_file = name_of_current_file_splitted[0]
    current_file_type = name_of_current_file_splitted[1]

    # Printing overview of all files in system
    print(f'\n\n{link_to_dat}')
    print(f" {'-' * 99}\n"
          f"{'%-100s' % f'|  Current file: {name_of_current_file}'}|\n"
          f"{'%-100s' % f'|  File type: {current_file_type}'}|\n"
          f"{not_csv_alerts}"
          f" {'-' * 99}\n")

    # Check for export call
    df_export(dfs_to_export=dfs_to_export,
              list_link_to_original_df_folder=link_to_folder,
              name_of_original_df=name_of_current_file)

    # Specifying the dependent
    target_index = 0
    while not target_name:
        target_ = input('What shall be the target?: ')
        if target_ in all_feature_names:
            target_name = target_
            target_index = all_feature_names.index(target_name)
            break
        print('Error: Target not found in system.')

    target_dtype = str(all_dtypes[target_index])
    dependent_mapping = ''
    if target_dtype == 'object':
        df[target_name] = df[target_name].astype('category')
        target_dtype = 'category'
        object_features.remove(target_name)
    if target_dtype == 'category':
        dependent_mapping = dict(enumerate(df[target_name].cat.categories))
        df[target_name] = df[target_name].astype('category').cat.codes
        if target_name in categorical_features:
            categorical_features.remove(target_name)
    if 'float' in target_dtype or 'int' in target_dtype:
        numerical_features.remove(target_name)

    print(f'\nTarget has been set as [{target_name}] for modeling.')
    if dependent_mapping:
        print(dependent_mapping)
    df_scatter_matrix(df)
    gc.collect()

    # *********************** FURTHER DATA TRUNCATING + PRELIMINARY GRAPHING FOR ANALYSIS *************************** #
    # Removing remaining object columns
    df.drop(object_features, axis=1, inplace=True)

    if boolean_standardize_all_numeric and not list_to_standardize:
        # Standardizing all numerical data
        scaler = StandardScaler()
        df.loc[:, numerical_features] = scaler.fit_transform(df[numerical_features])

    if list_to_standardize and not boolean_standardize_all_numeric:
        # Standardizing chosen numerical data
        scaler = StandardScaler()
        df.loc[:, list_to_standardize] = scaler.fit_transform(df[list_to_standardize])

    # Removing constant features
    df.drop(constant_features, axis=1, inplace=True)

    # II. Drop all rows with any NaN values
    number_of_nan_observations = np.count_nonzero(df.isnull().values.ravel())
    df.dropna(how='any', axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    current_number_of_observations = past_number_of_observations - number_of_nan_observations

    # Converting categorical columns into dummies
    print('\n***CONVERTING ALL CATEGORICALS INTO DUMMY FEATURES FOR MODELING')
    df, base_categories = vars_one_hot(df, categorical_features)

    current_all_columns_names = df.columns
    post_one_hot_number_of_features = len(current_all_columns_names)

    # III. Printing post-auto-truncating df characteristics
    print(f'\n{"-" * 80}')
    print(f'Previous number of features: {number_of_features}')
    print(f'Previous number of observations: {past_number_of_observations}')

    print(f'\n\t{len(object_features)} object features dropped: {object_features}')
    print(f'\t{len(constant_features)} constant features dropped: {constant_features}')
    print(f'\t{post_one_hot_number_of_features - number_of_features + len(object_features) + len(constant_features)} '
          f'dummy features added total, in which:'
          f'\n\t\t- {len(nan_features)} nan dummies already excluded belonging to {nan_features}')
    print(f'\n\t{number_of_nan_observations} rows with NaN values dropped')

    print(f'\nRemaining features:\t\t{post_one_hot_number_of_features}')
    print(f'Remaining observations:\t{current_number_of_observations}')
    print(f'{"-" * 80}')
    if post_one_hot_number_of_features > (current_number_of_observations / 10):
        print('**** WARNING: > 1 vars for every 10 observations ****')
    print()

    # IV. Scatter matrix graphing of df
    print(df)
    print(f'Base case: {base_categories}')
    print('\nPlease wait...Going to modeling...\n')

    target = df[target_name].copy()
    df.drop(target_name, axis=1, inplace=True)
    features = df
    return features, target


def df_scatter_matrix(df):
    """ Prints out scatter-matrix for the data with matplotlib """
    graph_vars_input = input('Scatter matrix the data? (y/n): ')
    if graph_vars_input == 'y' or graph_vars_input == 'Y':
        feature_name_to_be_colored = ''
        while not feature_name_to_be_colored:
            var_to_be_colored_input = input('Variable to be colored? (leave blank if none): ')
            if var_to_be_colored_input in df.columns:
                print(f'\nRecognizing ["{var_to_be_colored_input}"] as the variable to be colored.')
                feature_name_to_be_colored = var_to_be_colored_input
            elif not var_to_be_colored_input:
                print('No coloring variable. Default color for plots will be navy blue.')
                break
            else:
                print('Error: Variable not found in system.')

        # Color coded graphing (if applicable)
        transparency_alpha = 0.5
        coloring = 'navy'
        tuple_of_default_colors = ('black', 'red', 'blue', 'green', 'yellow', 'brown', 'purple')
        feature_to_be_colored = df[feature_name_to_be_colored].copy()
        if feature_name_to_be_colored and (len(feature_to_be_colored.unique()) < len(tuple_of_default_colors)):
            colors = dict(zip(feature_to_be_colored.unique(), tuple_of_default_colors))
            print(f'Color legend: {colors}')
            coloring = feature_to_be_colored.apply(lambda x: colors[x])
        elif len(df) > 60:
            transparency_alpha = 0.3
        print('\nDisplaying graphs...')

        # Creating and plotting the scatter_matrix
        pd.plotting.scatter_matrix(df, alpha=transparency_alpha, figsize=(8, 8), diagonal='hist',
                                   c='None', s=20, edgecolors=coloring,
                                   hist_kwds={'color': 'gray', 'ec': 'black', 'bins': 20, 'density': True})

        # Display settings and show plot
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close('all')


def df_set_row_as_header(df, header_row_index):
    """ set provided df's row as header """
    df.columns = df.iloc[header_row_index]
    df.drop(header_row_index, axis=0, inplace=True)


def vars_unpivot(df, vars_to_unpivot, unpivot_column_name='Unpivot', unpivot_value_column_name='Value'):
    """ unpivot df """
    df_unpivot = pd.melt(df,
                         id_vars=tuple(df.drop(vars_to_unpivot, axis=1).columns),
                         var_name=unpivot_column_name,
                         value_name=unpivot_value_column_name)
    return df_unpivot


def var_pivot(df, name_of_column_to_be_pivot):
    """ pivot df """
    df_pivot = pd.pivot(df, None, name_of_column_to_be_pivot)
    return df_pivot


def vars_one_hot(df, list_of_column_names):
    """ one hot categorical features (if provided) """
    base_categories = []
    for column in list_of_column_names:
        dummies = pd.get_dummies(df[column], prefix=column, dtype='int8')
        dummy_col_name_to_drop = dummies.columns[0]
        cat_dropped_name = dummy_col_name_to_drop.split(sep='_')[-1]
        base_categories.append(dummy_col_name_to_drop)
        dummies.columns = dummies.columns + f"/{cat_dropped_name}"
        dummy_col_name_to_drop = dummies.columns[0]

        dummies.drop(dummy_col_name_to_drop, axis=1, inplace=True)
        df = pd.concat([df, dummies], axis=1, copy=False)

    df.drop(list_of_column_names, axis=1, inplace=True)
    return df, base_categories


def generator_from_list(list_of_items_to_generate):
    for item_to_generate in list_of_items_to_generate:
        yield item_to_generate


def make_title(text_in_middle):
    """ make titles clearer and prettier """
    return f"{'<' * 80} {text_in_middle} {'>' * 100}"


def format_string_length(string, length, string_type='s'):
    string = ''.join(['%', str(length), string_type]) % string
    return string


def transform_main(x, list_to_drop, dict_of_poly, list_of_interactions,
                   all_interactions, all_poly_degrees):
    """ data transforms according to user's provided settings """
    error_check = False
    transformations_done = []

    if list_to_drop or dict_of_poly or list_of_interactions or all_interactions or \
            all_poly_degrees > 1:

        poly_error = bool
        interaction_error = bool
        dropping_error = bool

        # Add polynomials
        if dict_of_poly:
            poly_error = transform_add_poly(x, dict_of_poly)
            transformations_done.append('add poly')

        # Add ALL possible polinomials (bringing the equation to quadratic form, only works if polies are not specified)
        if not poly_error and all_poly_degrees > 1 and not dict_of_poly:
            transform_add_all_poly(x, all_poly_degrees)
            transformations_done.append('add all poly')

        # Add interactions
        if not poly_error and list_of_interactions:
            interaction_error = transform_add_interactions(x, list_of_interactions)
            transformations_done.append('add interactions')

        # Add ALL possible interactions (only if LIST_OF_INTERACTIONS is not specified)
        if not interaction_error and not poly_error and all_interactions and not list_of_interactions:
            transform_add_all_interactions(x)
            transformations_done.append('add all interactions')

        # Dropping specified variables
        if not interaction_error and not poly_error and list_to_drop:
            dropping_error = transform_drop_vars(x, list_to_drop)
            transformations_done.append('dropped custom variables')

        error_check = interaction_error and poly_error and dropping_error or False

    return error_check, transformations_done


def transform_add_interactions(x, list_of_interactions):
    """ data transform adding provided feature interactions """
    for interaction in list_of_interactions:
        try:
            splitted_interact = interaction.split(':')
            x[interaction] = x[splitted_interact[0]] * x[splitted_interact[1]]
        except KeyError:
            print('One of the interactions is invalid. The custom model will be terminated.')
            return True
    return False


def transform_add_all_interactions(x):
    """ data transform adding ALL feature interactions """
    original_x_len = len(x.columns)
    for var in range(0, original_x_len):
        for next_var in range(var + 1, original_x_len):
            x[f'{x.columns[var]}:{x.columns[next_var]}'] = \
                x[x.columns[var]] * x[x.columns[next_var]]


def transform_add_poly(x, dict_of_polies):
    """ data transform adding provided features with polynomial """
    if dict_of_polies:
        for feature_name, feature_degree in dict_of_polies.items():
            current_feature_name = feature_name
            for degree in range(2, feature_degree + 1):
                try:
                    x[f"{current_feature_name}^{degree}"] = x[current_feature_name] ** degree
                except KeyError:
                    print('One of the polies is invalid. The custom model will be terminated.')
                    return True
    return False


def transform_add_all_poly(x, degree):
    """ data transform adding polynomials for ALL existing features """
    original_x_len = len(x.columns)
    for var_index in range(0, original_x_len):
        for degree_level in range(2, degree + 1):
            x[f"{x.columns[var_index]}^{degree_level}"] \
                = x[x.columns[var_index]] ** degree_level


def transform_drop_vars(x, list_of_vars_to_drop):
    """ data transform dropping provided list of features """
    if list_of_vars_to_drop:
        try:
            x.drop(list_of_vars_to_drop, axis=1, inplace=True)
        except KeyError:
            print('One of the vars to drop is invalid. The custom model will be terminated.')
            return True
    return False


# LINEAR REGRESSION
def regression_main(x, y, cv, l1_ratio, penalty, individual_feature_analysis, quad_test_to_degree, display_graph):
    """ regression, main ops """
    x_with_const = sm.add_constant(x)

    # Linear regression for each individual variable (included interactions)
    individual_simple_reports = '(No individual reports)\n\n'
    if individual_feature_analysis:
        individual_simple_reports = regression_individuals_simple(x=x, y=y)

    model_simple = sm.OLS(y, x_with_const).fit()
    cv_mse_simple = regression_cv_simple(x=x, y=y, cv=cv)
    cv_mse_regular, params_regular, alpha_regular, l1_ratio_regular \
        = regression_cv_regular(x=x, y=y, cv=cv, l1_ratio=l1_ratio, penalty=penalty)

    print(f'{"-" * 80}\n[SIMPLE LINEAR REGRESSION WITH INDIVIDUAL FEATURE]')
    for individual_simple_report in individual_simple_reports:
        print(individual_simple_report)

    print(f'\n\n{"-" * 80}\n[MULTIPLE SIMPLE LINEAR REGRESSION]')
    print(model_simple.summary())
    print(f'\nMSE mean cross-validated: {str(round(cv_mse_simple, 4))}')
    print(f'RMSE cross-validated: {str(round(cv_mse_simple * 0.5, 4))}')

    print(f'\n\n{"-" * 80}\n[REGULARIZED LINEAR REGRESSION]')
    for key, value in dict(zip(x.columns, params_regular)).items():
        print(f'{key}: {value}')
    print(f'\nMSE mean cross-validated: {str(round(cv_mse_regular, 4))}')
    print(f'RMSE cross-validated: {str(round(cv_mse_regular * 0.5, 4))}')
    print(f'Alpha cross-validated: {str(round(alpha_regular, 4))}')
    print(f'L1 ratio cross-validated: {str(round(l1_ratio_regular, 4))}')

    # ------------------------ LINEAR REGRESSION: SOLVING TYPICAL LINEAR PROBLEMS -------------------------------- #
    print(f'\n\n{"-" * 80}\n[QUADRATIC TEST]')
    optimal_penalty = alpha_regular
    quad_report_1, quad_report_2, list_of_test_degrees, cv_mse_quad_simple_list, cv_mse_quad_regular_list = \
        regression_cv_quad_test(x=x, y=y, cv=cv, l1_ratio=l1_ratio,
                                optimal_penalty=optimal_penalty,
                                quad_test_to_degree=quad_test_to_degree)
    print(f'{quad_report_1}\n\n---Quad level with smallest value---\n{quad_report_2}')

    if display_graph:
        print(f'\n\n{"-" * 80}\n[TEST FOR POLY TREND, OUTLIERS, AND HIGH LEVERAGE OBSERVATIONS]\n')
        fitted_values_simple, leverage_simple, studentized_residual_simple, studentized_residual_color_simple = \
            regression_analysis_simple(model_simple, display_graph=display_graph)

        figs, axs = plt.subplots(2, 2, figsize=(8, 7))
        axs[0, 0].plot(list_of_test_degrees, cv_mse_quad_simple_list)
        axs[0, 0].title.set_text('Simple regression MSE (y) by degrees (x)')
        axs[1, 0].plot(list_of_test_degrees, cv_mse_quad_regular_list)
        axs[1, 0].title.set_text('Regularized regression MSE (y) by degrees (x)')
        # Finding possible poly trend and possible outliers (in red, studentized residual > 3 or < -3)
        axs[0, 1].scatter(fitted_values_simple, studentized_residual_simple, c='None',
                          edgecolors=studentized_residual_color_simple, s=20, alpha=0.3)
        axs[0, 1].title.set_text(f'LINEAR: Studentized residual (y) versus predicted value (x)')
        # Finding high leverage points (red points with significantly higher leverage than rest of data)
        axs[1, 1].scatter(leverage_simple, studentized_residual_simple, c='None',
                          edgecolors=studentized_residual_color_simple, s=20, alpha=0.3)
        axs[1, 1].title.set_text("LINEAR: Studentized residual (y) versus leverage (x)")
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close('all')


def regression_individuals_simple(x, y):
    """ regression simple """
    x_with_const = sm.add_constant(x)
    individual_simple_reports = []
    for current_var in x.columns:
        individual_model_simple = sm.OLS(y, x_with_const[['const', current_var]]).fit()
        current_var_p = individual_model_simple.pvalues[1]
        individual_simple_reports.append(f'Param regressed: {current_var}\n'
                                         f'Constant: {round(individual_model_simple.params[0], 8)}\n'
                                         f'Coefficient: {round(individual_model_simple.params[1], 8)}\n'
                                         f'Significance: {current_var_p < ALPHA} ({round(current_var_p, 8)})\n')
    return individual_simple_reports


def regression_cv_simple(x, y, cv):
    """ regression simple, cross-validated """
    cv_mse = 0
    np_x = x.values
    np_y = y.values

    kf = KFold(n_splits=cv)
    model_simple = LinearRegression()
    for train_index, test_index in kf.split(np_x):
        # Loop to get mse
        x_train, x_test = np_x[train_index], np_x[test_index]
        y_train, y_test = np_y[train_index], np_y[test_index]
        y_pred = model_simple.fit(x_train, y_train).predict(x_test)
        cv_mse += smte.mse(y_pred, y_test)
    cv_mse = cv_mse / (cv + 1)  # This works. Don't know why

    return cv_mse


def regression_cv_regular(x, y, cv, l1_ratio, penalty):
    """ regression cross validated with regularization """
    penalty_ = None
    if penalty != 0:
        penalty_ = np.full(shape=cv, fill_value=penalty)

    model_regularized = ElasticNetCV(cv=cv, l1_ratio=l1_ratio, alphas=penalty_)
    y_pred = model_regularized.fit(x, y).predict(x)
    cv_mse = smte.mse(y_pred, y)
    return cv_mse, model_regularized.coef_, model_regularized.alpha_, model_regularized.l1_ratio_


def regression_cv_quad_test(x, y, cv, l1_ratio, optimal_penalty, quad_test_to_degree):
    """ regression cross validated with quad test """
    if quad_test_to_degree >= 1:
        cv_mse_quad_simple_list = []
        cv_mse_quad_regular_list = []
        alpha_quad_regular_list = []
        l1_ratio_quad_regular_list = []

        list_of_test_degree = list(range(1, quad_test_to_degree + 1))
        for degree in list_of_test_degree:
            quad_test_x = x.copy()
            transform_add_all_poly(x=quad_test_x, degree=degree)
            cv_mse = regression_cv_simple(x=quad_test_x, y=y, cv=cv)
            cv_mse_regular, params_regular, alpha_regular, l1_ratio_regular \
                = regression_cv_regular(x=quad_test_x, y=y, cv=cv, l1_ratio=l1_ratio, penalty=optimal_penalty)
            cv_mse_quad_simple_list.append(cv_mse)
            cv_mse_quad_regular_list.append(cv_mse_regular)
            alpha_quad_regular_list.append(alpha_regular)
            l1_ratio_quad_regular_list.append(l1_ratio_regular)

        report = pd.DataFrame(data={'cv_mse_simple': cv_mse_quad_simple_list,
                                    'cv_mse_regular': cv_mse_quad_regular_list,
                                    'alpha_regular': alpha_quad_regular_list,
                                    'l1_ratio_regular': l1_ratio_quad_regular_list},
                              index=list_of_test_degree)
        report2 = report.idxmin(axis=0, skipna=True)

        return report, report2, list_of_test_degree, cv_mse_quad_simple_list, cv_mse_quad_regular_list


def regression_analysis_simple(model_simple, display_graph):
    """ Analysis of simple regression to find outliers """
    if display_graph:
        studentized_residual_simple = model_simple.get_influence().resid_studentized_external
        fitted_values_simple = model_simple.fittedvalues
        leverage_simple = model_simple.get_influence().hat_matrix_diag
        studentized_residual_color_simple = np.where(studentized_residual_simple < -3,
                                                     'red',
                                                     np.where(studentized_residual_simple > 3,
                                                              'red',
                                                              'navy'))
        print(f'Outlier observations (indices): {np.where(studentized_residual_color_simple == "red")[0]}')
    else:
        fitted_values_simple, leverage_simple, studentized_residual_simple, studentized_residual_color_simple \
            = None, None, None, None

    return fitted_values_simple, leverage_simple, studentized_residual_simple, studentized_residual_color_simple


# CLASSIFICATION
def classification_main(x, y, cv, list_of_class_weights, interpretation_with_sm, individual_feature_analysis,
                        quad_test_to_degree, n_neighbors, display_graph):
    """ classification, main ops """
    # Weight converting
    weights_dict = classification_assigning_weights(y=y, list_of_class_weights=list_of_class_weights)

    # Model initiations
    model_one_vs_rest = LogisticRegression(multi_class='ovr',
                                           class_weight=weights_dict,
                                           solver='saga')
    model_multi = LogisticRegression(multi_class='multinomial',
                                     class_weight=weights_dict,
                                     solver='saga')
    model_ordinal = LogisticAT(alpha=1)  # alpha = 0 means no regularization
    model_knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

    model_list = [model_one_vs_rest, model_multi, model_ordinal, model_knn]
    list_of_model_name = ['Logistic OVR', 'Logistic Multinomial', 'Logistic Ordinal', 'K-nearest neighbors']

    # Simple logistic regression for each class with each feature (marginal effects, std err, and p-values included)
    zero_dict_of_reports, zero_dict_of_train_error_rates = None, None
    if individual_feature_analysis and interpretation_with_sm:
        zero_dict_of_reports, zero_dict_of_train_error_rates = \
            classification_logit_ovr_with_sm_with_each_feature(x=x, y=y)

    # Simple logistic regression for each class with all features (marginal effects, std err, and p-values included)
    first_dict_of_reports, first_dict_of_train_error_rates = None, None
    if interpretation_with_sm:
        first_dict_of_reports, first_dict_of_train_error_rates = classification_logit_ovr_with_sm(x=x, y=y)

    # Multinomial logistic regression (marginal effects, std err, and p-values included)
    second_report, second_train_acc_rate, second_train_pred_target = None, None, None
    if interpretation_with_sm:
        second_report, second_train_acc_rate, second_train_pred_target = classification_logit_mn_with_sm(x=x, y=y)

    # Quadratic test for each model
    third_report, confusion_matrix_dict = classification_cv_models_quad_test(
        x=x, y=y, model_list=model_list, list_of_model_name=list_of_model_name,
        cv=cv, quad_test_to_degree=quad_test_to_degree)

    # -------------------------------------REPORT------------------------------------
    if zero_dict_of_reports:
        print(f"[LOGISTIC BINARY WITH EACH CLASS WITH EACH FEATURE]")
        for key, value in zero_dict_of_reports.items():
            print(f'[{key}]')
            for value1 in value:
                print(f'{value1}\n')
        print('---ACCURACY RATE RESULTS---')
        for key, value in zero_dict_of_train_error_rates.items():
            print(f'{key}: {value}')

    if first_dict_of_reports:
        print(f"\n\n[LOGISTIC BINARY WITH EACH CLASS AND ALL FEATURES]")
        for key, value in first_dict_of_reports.items():
            print(f'[{key}]\n{value}\n')
        print('---ACCURACY RATE RESULTS---')
        for key, value in first_dict_of_train_error_rates.items():
            print(f'{key}: {value}')

    if second_train_acc_rate:
        print('\n\n[LOGISTIC MULTINOMIAL WITH ALL CLASSES]')
        print(f'{second_report}\n\nTrain accuracy rate: {second_train_acc_rate}')

    print('\n\n[CLASSIFICATION MODELS AND QUAD TEST]: CROSS-VALIDATED ACCURACY RATES BY QUADRATIC LEVEL')
    print(third_report)
    print(f'Model and quad level with maximum accuracy rate: '
          f'{third_report.stack().index[np.argmax(third_report.values)]}')

    print('\n\n[CONFUSION MATRICES]: TRAIN PREDICTION VS ACTUAL VALUES FOR FIRST QUADRATIC LEVEL')
    print(weights_dict)
    for key, value in confusion_matrix_dict.items():
        print(f'---{key}---\n{value}\n')
    if display_graph:
        third_report.T.plot()
        plt.show()


def classification_assigning_weights(y, list_of_class_weights):
    """ classification, assigning weights to different classes of the target """
    labels_list = y.unique()
    if list_of_class_weights and len(list_of_class_weights) == len(labels_list):
        class_weight = dict(zip(labels_list, list_of_class_weights))
    else:
        class_weight = 'balanced'
    return class_weight


def classification_logit_ovr_with_sm(x, y):
    features = sm.add_constant(x)
    all_features = features.columns[1:]
    all_classes = y.unique()

    list_of_report, list_of_train_error_rate = [], []

    for label in all_classes:
        target = np.where(y == label, 1, 0)

        try:
            logit = sm.Logit(target, features).fit(disp=False)
        except sm.tools.sm_exceptions.PerfectSeparationError:
            print('PerfectSeparationError for classification_logit_ovr')
            logit = sm.Logit(target, features).fit(method='bfgs', disp=False)
        except np.linalg.LinAlgError:
            print('np.linalg.LinAlgError for classification_logit_ovr')
            return None, None

        marginal_effects = logit.get_margeff()
        pred_target = np.asarray(logit.predict(features).astype(int))
        train_error_rate = metrics.accuracy_score(target, pred_target)

        data = {'Marginal Effects': np.round(marginal_effects.margeff, 4),
                'Std err': np.round(marginal_effects.margeff_se, 4),
                'p-values': np.round(marginal_effects.pvalues, 4)}
        report = pd.DataFrame(data=data, index=all_features)
        list_of_report.append(report)
        list_of_train_error_rate.append(round(train_error_rate, 4))

    all_report = dict(zip(all_classes, list_of_report))
    train_error_rate = dict(zip(all_classes, list_of_train_error_rate))

    return all_report, train_error_rate


def classification_logit_ovr_with_sm_with_each_feature(x, y):
    features = sm.add_constant(x)
    all_features = features.columns[1:]
    all_classes = y.unique()

    class_report, class_train_error_rate = [], []

    for label in all_classes:
        target = np.where(y == label, 1, 0)
        list_of_reports, list_of_train_error_rate = [], []
        for feature in all_features:
            try:
                logit = sm.Logit(target, features[['const', feature]]).fit(disp=False)
            except sm.tools.sm_exceptions.PerfectSeparationError:
                print('PerfectSeparationError for classification_logit_ovr')
                logit = sm.Logit(target, features[['const', feature]]).fit(method='bfgs', disp=False)
            except np.linalg.LinAlgError:
                print('np.linalg.LinAlgError for classification_logit_ovr')
                return None, None

            marginal_effects = logit.get_margeff()
            pred_target = np.asarray(logit.predict(features[['const', feature]]).astype(int))
            train_error_rate = metrics.accuracy_score(target, pred_target)

            data = {'Marginal Effects': np.round(marginal_effects.margeff, 4),
                    'Std err': np.round(marginal_effects.margeff_se, 4),
                    'p-values': np.round(marginal_effects.pvalues, 4)}
            report = pd.DataFrame(data=data, index=[feature])
            list_of_reports.append(report)
            list_of_train_error_rate.append(round(train_error_rate, 4))
        class_report.append(list_of_reports)
        class_train_error_rate.append(dict(zip(all_features, list_of_train_error_rate)))

    master_rep = pd.concat(class_report, axis=0, keys=all_classes)
    print(master_rep)

    all_report = dict(zip(all_classes, class_report))
    train_error_rate = dict(zip(all_classes, class_train_error_rate))

    return all_report, train_error_rate


def classification_logit_mn_with_sm(x, y):
    features = sm.add_constant(x)
    all_classes = y.unique()
    all_features = features.columns[1:]

    try:
        mnlogit = sm.MNLogit(y, features).fit(disp=False)
    except sm.tools.sm_exceptions.PerfectSeparationError:
        print('PerfectSeparationError for classification_logit_mn')
        mnlogit = sm.MNLogit(y, features).fit(method='bfgs', disp=False)

    train_pred_target = list(pd.DataFrame(mnlogit.predict(features)).idxmax(axis=1))
    try:
        for index in range(len(train_pred_target)):
            train_pred_target[index] = all_classes[train_pred_target[index]]
        train_acc_rate = round(metrics.accuracy_score(y, train_pred_target), 4)
    except IndexError:
        print('ERROR: Check classification_logit_mn')
        return None, None, None

    data = dict(zip([f'mff_{item}' for item in all_classes],
                    np.round(np.transpose(mnlogit.get_margeff().margeff), 4)))
    data2 = dict(zip([f'stdErr_{item}' for item in all_classes],
                     np.round(np.transpose(mnlogit.get_margeff().margeff_se), 4)))
    data3 = dict(zip([f'pvalues_{item}' for item in all_classes],
                     np.round(np.transpose(mnlogit.get_margeff().pvalues), 4)))

    data.update(data2)
    data.update(data3)
    report = pd.DataFrame(data=data, index=all_features)

    return report, train_acc_rate, train_pred_target


def classification_cv_models_quad_test(x, y, model_list, list_of_model_name,
                                       cv, quad_test_to_degree):
    confusion_matrix_list = []
    poly_error_list = []

    tuple_of_test_degree = tuple(range(1, quad_test_to_degree + 1))
    quad_test_y = y.copy()

    for degree in tuple_of_test_degree:
        cv_error_list = []
        quad_test_x = x.copy()
        transform_add_all_poly(x=quad_test_x, degree=degree)

        for test_model in model_list:
            # Creating confusion matrix for degree 1
            if degree == 1:
                fitted_model = test_model.fit(X=quad_test_x, y=quad_test_y)
                confusion_matrix_list.append(classification_confusion_matrix(
                    labels=quad_test_y,
                    pred_labels=np.round(fitted_model.predict(quad_test_x), 0),
                    all_names_not_categorized=y.unique().astype('object')))

            # Main loop
            cv_error_list.append(np.round(np.mean(cross_val_score(estimator=test_model, X=quad_test_x,
                                                                  y=quad_test_y, cv=cv, scoring='accuracy')), 4))

        poly_error_list.append(dict(zip(list_of_model_name, cv_error_list)))

    report = pd.DataFrame(dict(zip(tuple_of_test_degree, poly_error_list)))
    confusion_matrix_dict = dict(zip(list_of_model_name, confusion_matrix_list))

    return report, confusion_matrix_dict


def classification_confusion_matrix(labels, pred_labels, all_names_not_categorized):
    all_labels = labels.unique()
    confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_true=labels, y_pred=pred_labels,
                                                             labels=all_labels),
                                    columns=all_names_not_categorized, index=all_names_not_categorized)
    confusion_matrix.loc['Pred_total', :] = confusion_matrix.sum(axis=0)
    confusion_matrix.loc[:, 'True_total'] = confusion_matrix.sum(axis=1)
    for column in confusion_matrix.columns:
        confusion_matrix[column] = confusion_matrix[column].astype(int)
    return confusion_matrix


# ***************** MAIN CODE ******************************** #
if __name__ == '__main__':
    main(link_to_dat=LINK_TO_DAT,
         excel_sheet_index=EXCEL_SHEET_INDEX,
         frac=FRAC,
         all_feature_types=ALL_FEATURE_TYPES,
         list_to_standardize=LIST_TO_STANDARDIZE,
         boolean_standardize_all_numeric=BOOLEAN_STANDARDIZE_ALL_NUMERIC,
         list_to_datetime=LIST_TO_DATETIME,
         regression=REGRESSION, list_to_drop=LIST_TO_DROP, dict_of_poly=DICT_OF_POLIES,
         list_of_interactions=LIST_OF_INTERACTIONS,
         all_interactions=ALL_INTERACTIONS, all_poly_degrees=ALL_POLY_DEGREES,
         cv=CV, l1_ratio=L1_RATIO, penalty=PENALTY,
         quad_test_to_degree=QUAD_TEST_TO_DEGREE, display_graph=DISPLAY_GRAPH,
         classification=CLASSIFICATION, list_of_class_weights=LIST_OF_CLASS_WEIGHTS,
         individual_feature_analysis=INDIVIDUAL_FEATURE_ANALYSIS, n_neighbors=N_NEIGHBORS)
