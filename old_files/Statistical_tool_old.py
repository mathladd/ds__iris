import statsmodels.api as sm
import statsmodels.tools.eval_measures as smte
import statsmodels.tools.sm_exceptions as sm_except

from sklearn.exceptions import ConvergenceWarning
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.linear_model import LinearRegression, LogisticRegression

from mord import LogisticAT

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
import matplotlib.pyplot as plt

import os
import warnings

# ************************************************** SETUPS ******************************************************* #
pd.set_option('display.expand_frame_repr', False)

plt.rcParams.update({'font.size': 8})
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=sm_except.ConvergenceWarning)


# 1. ********************************************* MASTER KEYS **************************************************** #
"""MASTER SETTINGS"""
# r'C:\Users\CB495EF\Downloads\self projects\data\Kaggle\NYC 2018 Yellow Taxi Trip\2018_Yellow_Taxi_Trip_Data.csv'
# r'C:\Users\CB495EF\PycharmProjects\MoMo\data4\Mercury vol2 Q1.csv'
# r'C:\Users\CB495EF\PycharmProjects\MoMo\data4\Mercury vol2 Q3.csv'
# r'C:\Users\CB495EF\Downloads\self projects\data\Kaggle\Iris\iris.txt'
CURRENT_FILE_LINK = r'C:\Users\CB495EF\Downloads\self projects\data\Kaggle\Iris\iris.txt'
INDEX_SHEET = 0  # FOR EXCEL FILE ONLY, set to SHEET INDEX of current excel file
FRAC = 1  # OPTIMIZATION: Get sample of a fraction of the data
VARIABLE_TYPES = {

}  # OPTIMIZATION: Specify column types as a dictionary (e.g. {'a': np.float64} to reduce memory usage


"""MODEL SETTINGS"""
DEPENDENT = ''
ALPHA = 0.05
K_FOLD_CV = 10  # Set to 1 to eliminate cross validation. Set to n to do LOOCV
PENALTY = 1  # Penalty for lasso and/or Ridge regularization

LIST_OF_INTERACTIONS = [

]  # List of all interactions to add e.g. ['Age:Fare', 'Age:Health']
LIST_OF_POLY_VARS = [

]  # List of variables to add degree vars e.g. ['Age', 'Fare']
LIST_OF_DEGREE = [

]  # List of degrees for each of the variables above, matched by indices e.g. [2, 5]
LIST_TO_DROP = [

]  # List of variables to be dropped from the linear model e.g. ['Unnamed: 0', 'col20']

ALL_INTERACTIONS = False  # Set this to False unless you know what you are doing !!
ALL_POLY = False  # Set this to False unless you know what you are doing !!
ALL_POLY_DEGREE = 2
QUADRATIC_TEST_TO_DEGREE = 2


# Linear regression
LINEAR = True
QUADRATIC_TEST_LINEAR = True  # List of poly levels to perform MSE mining
LASSO_OR_RIDGE = 1  # Set this to 0 for linear ridge regression


# Classification
CLASSIFICATION = True
LIST_OF_WEIGHTS = []
N_NEIGHBORS = 10  # Number of neighbors for K-nearest neighbors classifier


# ***************** FUNCTION DEFINITIONS ******************************** #
def proc_processing(df):
    # 1. DATA CLEANING
    #  *** Set headers as row   --> set_headers_as_row(df, index_of_to_be_header_row)
    #  *** Renaming all columns --> df.columns = ['name1', 'name2', 'name3']
    #  *** Unpivot              --> df = unpivot(df, list_of_var_names_to_be_unpivot,
    #                                                unpivot_new_column_name, unpivot_value_column_name)
    #  *** Pivot                --> df = pivot(df, name_of_column_to_be_pivot)
    #  *** Transpose            --> df = df.transpose()
    #  *** Drop variables       --> df.drop(list_of_var_names_to_exclude, axis=1, inplace=True)
    #  *** Drop rows            --> df.drop(list_row_indices, axis=0, inplace=True) note:(df.reset_index() first)
    #  *** Replace values (e.g. NaN) in a column --> df[column_name] = df[column_name].replace(list_of_original_values,
    #                                                                                               List_of_new_value)
    #  ***replace NaN w/ the mean                           --> df['a'].fillna(df['a'].mean())
    #                 w/ forward fill                       --> df['a'].fillna(method='ffill')
    #                 w/ each mean of 'a' grouped by 'b'    --> df['a'].fillna(df.groupby('b')['a'].transform('mean'))
    #                 w/ the nearest last value             --> df.fillna(method='ffill', inplace=True)
    #                 w/ the nearest next value             --> df.fillna(method='bfill', inplace=True)
    #  ***Convert var to numerical category type            --> df = var_to_numerical_cat(df, column_name_to_convert)

    # 2. EVALUATION
    #  *** Sort df by col1 ascending, then col2 descending --> df.sort_values([col1,col2],ascending=[True,False])
    #  *** Change column by condition, similar to excel's IF statement
    #                   --> df[name_of_column_to_be_changed] = np.where(
    #                                                          df[name_of_conditional_column] == value_to_sort_by,
    #                                                          value_if_condition_matched,
    #                                                          value_if_condition_not_matched)
    #      - Acceptable: name_of_conditional_column = name_of_column_to_be_changed
    #      - Acceptable: value_if_condition_matched = calculation
    #  *** Grouping df by 'Sex', then by 'Age', aggregated through AVERAGING 'Pclass' & AVERAGING 'Survived'
    #                                               --> df = df.groupby(['Sex','Age'], as_index=False).agg(
    #                                                       {'Pclass': lambda x: x.mean(), 'Survived': np.mean()})
    #  *** Apply function f to df (can also use apply as .agg replacement) --> df = df.apply(f)
    #  *** Split qualitative column into many columns with [0, 1] values only (must remove NaN first)
    #                                               --> df = one_hot(df, column_name_to_split)
    #  *** Merge df1 & 3 columns of df2, matching by customer ID (df1's Cust_ID matched df2's ID)
    #                                               --> df = pd.merge(
    #                                               df1, df2.ix[:,["ID","Revenue","Expenses"]],
    #                                               left_on="Cust_ID", right_on="ID"
    #                                               )
    #  *** Merge a List of dfs (horizontally)       --> mergedDf = pd.concat(dfList, axis=1)
    #  *** Append a List of dfs (vertically)        --> mergedDf = pd.concat(dfList, axis=0)

    # 3. DATA CALCULATION
    #  *** Select first row                         --> df.iloc[[0]]
    #  *** Select rows with c1 > 0.5 && c2 < 0.7    --> df[(df[c1] > 0.5) & (df[c2] < 0.7)]

    """1. DATA CLEANING: PIVOT/UNPIVOT, dealing with NaN/REDUNDANCIES, MERGING/CONCATENATING, adjusting INCONSISTENCY"""
    df_reset_index(df, True)
    # ------------------------- Code here ------------------------- #
    #

    #
    # ----------------------- End code here ----------------------- #

    """2. EVALUATION: QUALITATIVE adjustments, CALCULATED COLUMNS, GROUP BY"""
    proc_analysis(df, auto_convert_var_dtypes=True)
    # Printing raw data
    # ------------------------- Code here ------------------------- #
    #

    #
    # ----------------------- End code here ----------------------- #

    """3. DATA CALCULATION: MAKING AND PRINTING CUSTOM CALCULATIONS"""
    # ------------------------- Code here ------------------------- #
    #

    #
    # ----------------------- End code here ----------------------- #

    dfs_to_export = []  # Add dfs to be exported here (into individual sheets for xlsx or files for csv)
    return df, dfs_to_export


# II. Data input/output functions
def proc_importing(current_file_link, index_of_sheet, frac, variable_types):
    print(f'\nReading: {current_file_link}')
    unique_file_notice_string = ''
    file_size = os.stat(current_file_link).st_size
    if not variable_types:
        variable_types = None

    if current_file_link.find('.csv') != -1 or current_file_link.find('.txt') != -1:
        if file_size > 1000000000:
            print('File is too large. Please go to def import_df to perform MapReduce with Dask.')
            dask_df = dd.read_csv(current_file_link, dtype=variable_types)
            # read multiple files: dask_dfs = dd.read_csv('data_path/2014-*.csv')
            # export dask_df to single csv: dask_df.to_csv('path/to/csv.csv', single_file=True)
            print(f'\nNumber of variables: {len(dask_df.columns)}')
            print(f'Sample:\n{dask_df.head()}')
            print('\nCreating the Client...')

            client = Client()

            print('Client created. Map reducing initialized...')

            if frac < 1:
                dask_df = dask_df.sample(frac=frac)
                print(f'Returning {int(frac*100)}% of data')

            # -------------------------------- MAP REDUCING STARTS --------------------------------------
            # dask_df = dask_df[['NETWORK', CHANNEL]]
            # dask_df = dask_df.drop(["NETWORK", "INTERNAL_SEGMENTATION", "BANK_OR_UNBANK"], axis=1)
            # dask_df = dask_df.groupby(['GEOGRAPHY', 'BANK_VISA']).agg({'UNIQUE_USER': 'sum'}).reset_index()
            # dask_df = dask_df.assign(Z = dask_df["X"] + dask_df["Y"]) --> Add calculated col
            # dask_df = dask_df[dask_df["X"] > 0]                       --> Choose only rows with value in col "X" > 0
            # dask_df = dask_df.sample(frac=0.005)                      --> Sampling the data for faster loading
            #
            #

            #
            #
            # -------------------------------- MAP REDUCING ENDS ----------------------------------------
            dask_df = client.persist(dask_df)  # Store df into cluster's memory (note: cluster must have enough RAM)
            print('Finished map reducing. Printing additional computations...')
            # ---------------------------- CUSTOM HIGH-LEVEL COMPUTATIONS -----------------------------
            # print(dask_df['NUMBERS_USER'].sum().compute())
            #

            #
            #
            # -------------------------------- COMPUTATIONS ENDS ----------------------------------------
            print('[Ended high-level query]')
            df = dask_df.compute().reset_index(drop=True)  # Store df into single machine's RAM !!
            print('Closing Client...')
            client.close()
        else:
            df = pd.read_csv(current_file_link, dtype=variable_types)

    elif (current_file_link.find('.xlsx') or current_file_link.find('.xls')) != -1:
        df = pd.read_excel(current_file_link, sheet_name=index_of_sheet, dtype=variable_types)
        list_of_sheet_names = pd.ExcelFile(current_file_link).sheet_names
        unique_file_notice_string = \
            f"{'%-100s' % f'|  Displaying sheet: {str(index_of_sheet + 1)}/{len(list_of_sheet_names)}'}|\n"\
            f"{'%-100s' % f'|  List of sheets: {str(list_of_sheet_names)}'}|\n"

    elif current_file_link.find('.hdf') != -1:
        df = pd.read_hdf(current_file_link, key='test', dtype=variable_types)

    else:
        print('Cannot determine type of file')
        df = pd.DataFrame

    return df, unique_file_notice_string


def proc_exporting(df, dfs_to_export, link_to_folder, name_of_current_file):
    export_input = input('Export df to csv or xlsx? (csv/xlsx): ')
    if export_input == 'csv':
        print('\nExporting...')
        if not dfs_to_export:
            current_csv_file_link = link_to_folder.copy()
            current_csv_file_name = f'{name_of_current_file} (processed).csv'
            current_csv_file_link.append(current_csv_file_name)
            export_link = '\\'.join(current_csv_file_link)
            df.to_csv(export_link, index=False)
            print(f'Exported: {current_csv_file_name}')
        else:
            csv_index = 0
            while csv_index < len(dfs_to_export):
                current_csv_file_link = link_to_folder.copy()
                current_csv_file_name = f"{name_of_current_file} REPORT {str(csv_index)}.csv"
                current_csv_file_link.append(current_csv_file_name)
                export_link = '\\'.join(current_csv_file_link)
                dfs_to_export[csv_index].to_csv(export_link, index=False)
                print(f'Exported: {current_csv_file_name}')
                csv_index += 1
        print()

    elif export_input == 'xlsx':
        print('\nExporting...')
        if not dfs_to_export:
            current_xlsx_file_link = link_to_folder.copy()
            current_xlsx_file_name = f'{name_of_current_file} (processed).xlsx'
            current_xlsx_file_link.append(current_xlsx_file_name)
            export_link = '\\'.join(current_xlsx_file_link)
            df.to_excel(excel_writer=export_link, sheet_name=name_of_current_file, index=False)
            print(f'Exported: {current_xlsx_file_name}')
        else:
            current_xlsx_file_link = link_to_folder.copy()
            current_xlsx_file_name = f'{name_of_current_file} SHEETS_OF_REPORTS.xlsx'
            current_xlsx_file_link.append(current_xlsx_file_name)
            export_link = '\\'.join(current_xlsx_file_link)
            dfs_to_export[0].to_excel(excel_writer=export_link, sheet_name='REPORT 0', index=False)
            export_sheet_index = 1
            while export_sheet_index < len(dfs_to_export):
                with pd.ExcelWriter(export_link, mode='a') as writer:
                    dfs_to_export[export_sheet_index]\
                        .to_excel(writer, sheet_name=f'REPORT {str(export_sheet_index)}', index=False)
                export_sheet_index += 1
            print(f'Exported: {current_xlsx_file_name}')

        print()


def proc_display_raw(df):
    print(f'[RAW DATA FRAME DISPLAY]\n{df}\n[END OF RAW]\n')


def proc_analysis(df, auto_convert_var_dtypes=True):
    """
    This function displays the characteristics of the df being passed into, as well as automatically coverting
    categorical and date columns into their respective correct dtype

    !: Problems
        (   1  = the variable is of OBJECT-type
            10 = the variable has NaN values
            11 = the variable both is of OBJECT-type and has NaN values
            0  = means the variable is good for modeling    )

    ID: The name of each variable
    uniques: Number of uniques in each variable

    sorted: If the uniques displayed are sorted or not (either by date, numerical, or alphabetical)
        (   1   = the variable is sorted by numpy
            10  = the variable is converted to string, then sorted alphabetically by numpy
            0   = the variable is NOT SORTED    )

    uniques_show/uniq_show: display of uniques
    type: data type of the variable
    %_NA: % of NaN values in the variable (%)
    Mod: the mod of the variable (or most frequent item)
    Md_F: Frequency of Mod in % (%)
    Other statistical displays

    :param df: pd.DataFrame. The df to display characteristics
    :param auto_convert_var_dtypes: Boolean. Convert object column to category and date if True

    :return: None. Auto-print df characteristics
    """

    all_df_columns_names = tuple(df.columns)
    len_df = len(df)
    all_dtypes = df.dtypes
    unique_display_width = 150
    unique_display_half = int(unique_display_width / 2)

    nan_counter = 0
    object_counter = 0
    all_vars_char = []
    all_vars_uniques = []

    # Looping through each variable
    column_index = 0
    while column_index < len(all_df_columns_names):
        current_df_column = str(all_df_columns_names[column_index])
        all_uniques = df[current_df_column].unique()
        num_of_unique = len(all_uniques)

        flag_problem = 0
        flag_int = False
        flag_float = False
        flag_date = False
        flag_no_time = False

        # Sorting out data types and looking for object variables
        current_column_dtype = all_dtypes[column_index]
        if current_column_dtype in ('int64', 'int32'):
            flag_int = True
        elif current_column_dtype in ('float64', 'float32'):
            flag_float = True
        elif current_column_dtype == 'datetime64[ns]':
            current_column_dtype = 'dt64'
            flag_date = True
        elif current_column_dtype == 'object':
            try:
                df[current_df_column] = df[current_df_column].astype('datetime64[ns]')
                current_column_dtype = 'dt64'
                flag_date = True
            except ValueError:
                if num_of_unique < int(len_df/2) and auto_convert_var_dtypes:
                    df[current_df_column] = df[current_df_column].astype('category')
                    current_column_dtype = 'cat'
                    all_dtypes[current_df_column] = 'category'
                else:
                    object_counter += 1
                    flag_problem += 1

        # Looking for variables containing NaN
        if df[current_df_column].isnull().any():
            percent_nan = str(round(((df[current_df_column].isna().sum()) / len_df) * 100, 1))
            nan_counter += 1
            flag_problem += 10
        else:
            percent_nan = ''

        # Basic statistics + distribution for quantitative, and the mode and frequency percent for qualitative data
        distribution = ''
        sum_num = ''

        # CHECK FOR INTEGER DATA TYPE
        if flag_float or flag_int:
            # Sum and descriptions
            sum_num = df[current_df_column].sum()
            describe = list(df[current_df_column].describe())
            describe_mode = df[current_df_column].mode()[0]
            describe_mode_freq = df.loc[
                df[current_df_column] == describe_mode, current_df_column
            ].count()
            describe_mode_freq_percent = round((describe_mode_freq / len_df) * 100, 1)

            # Converting integer descriptions
            if flag_int:
                for index in range(3, 8):
                    describe[index] = int(describe[index])

            # Summarize description
            description = (describe_mode, describe_mode_freq_percent, describe[1], describe[2],
                           describe[3], describe[4], describe[5], describe[6], describe[7])

            # Distribution of int and float data types
            if describe[7] != describe[3]:
                distribution_25 = int(round(((describe[4] - describe[3]) / (describe[7] - describe[3])) * 24, 0))
                distribution_50 = int(round(((describe[5] - describe[3]) / (describe[7] - describe[3])) * 24, 0))
                distribution_75 = int(round(((describe[6] - describe[3]) / (describe[7] - describe[3])) * 24, 0))
                distribution_list = ['-'] * 24
                distribution_list.insert(distribution_50, '|')
                distribution_list.insert(distribution_25, '(')
                distribution_list.insert(distribution_75 + 2, ')')
                distribution = ''.join(distribution_list)
            else:
                distribution = '-' * (24 + 3)

        elif flag_date:
            describe = list(df[current_df_column].describe())
            describe_mode = describe[2]
            describe_mode_freq_percent = round(describe[3] / len_df * 100, 1)
            min_date = describe[4]
            max_date = describe[5]
            if len(df[current_df_column][pd.notnull(df[current_df_column])]
                    .dt.hour.unique()) == 1:
                flag_no_time = True
            if flag_no_time:
                describe_mode = str(describe_mode.date())
                min_date = str(min_date.date())
                max_date = str(max_date.date())
            description = (describe_mode, describe_mode_freq_percent, '', '', min_date, '', '', '', max_date)

        # object or categorical types
        else:
            describe = list(df[current_df_column].describe())
            describe_mode = str(describe[2])
            describe_mode_freq_percent = round(describe[3] / len_df * 100, 1)
            description = (describe_mode, describe_mode_freq_percent, '', '', '', '', '', '', '')

        # Unique values for each variable
        uniques_sorted = 1
        try:
            all_uniques = np.sort(all_uniques)
        except TypeError:
            try:
                all_uniques = all_uniques.astype(str)
                all_uniques = np.sort(all_uniques)
                uniques_sorted += 10
            except TypeError:
                uniques_sorted = 0

        # Display date only if no time
        if flag_no_time:
            all_uniques = tuple(all_uniques[all_uniques != 'nan'].astype('datetime64[D]').astype(str))
        else:
            all_uniques = tuple(all_uniques)
        all_uniques_as_string = str(all_uniques)

        # Creating unique show
        if len(all_uniques_as_string) < unique_display_width:
            unique_show = f'{all_uniques_as_string[:unique_display_half-5]}' \
                          f'{all_uniques_as_string[unique_display_half-5:]}'
        else:
            unique_show = f'{all_uniques_as_string[:unique_display_half-5]} ... ' \
                          f'{all_uniques_as_string[-unique_display_half+5:]}'

        unique_show_sample = str(all_uniques[0])[:20]

        # Summarizing each variable
        each_variable_char_uniques = (flag_problem, current_df_column, num_of_unique, uniques_sorted,
                                      ('%-' + str(unique_display_width-5) + 's') % unique_show)
        each_variable_char = (flag_problem, current_df_column, num_of_unique,
                              unique_show_sample, current_column_dtype, percent_nan,
                              description[0], float(description[1]), sum_num,
                              description[2], description[3], description[4],
                              description[5], description[6], description[7], description[8], distribution)

        # Adding summaries to reports
        all_vars_uniques.append(each_variable_char_uniques)
        all_vars_char.append(each_variable_char)
        column_index += 1

    report_uniques = pd.DataFrame(data=all_vars_uniques,
                                  columns=('!', 'ID', 'uniques', 'sorted', 'uniques_show'))
    report = pd.DataFrame(data=all_vars_char,
                          columns=('!', 'ID', 'uniques', 'uniq_sample', 'type', '%_NA', 'Mod', 'Md_F',
                                   'Sum', 'Mean', 'std', 'Min', '0.25', 'Med', '0.75', 'Max', 'Distribution'))

    # REPORT
    print(f'[DATA FRAME POST-CLEANING]\n{df}\n[END OF DATA FRAME POST-CLEANING DISPLAY]')
    print(f"\nDATA VARIABLES SUMMARY: {len_df} observations and {len(all_df_columns_names)} variables")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           'display.max_colwidth', unique_display_width):
        print(report_uniques)
    print('-' * 200)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(report)
    print(f'\nThere are {object_counter} OBJECT-type variables')
    print(f'There are {nan_counter} variables containing NaN values')
    print(f'All dtypes: {dict(zip(all_df_columns_names, all_dtypes.astype(str)))}')
    print('\n' + '\t' * 12 + '---End---')


def proc_scatter_matrix(df):
    graph_vars_input = input('Scatter matrix the data? (y/n): ')
    if graph_vars_input == 'y' or graph_vars_input == 'Y':
        var_to_be_colored = ''
        while not var_to_be_colored:
            var_to_be_colored_input = input('Variable to be colored? (leave blank if none): ')
            if var_to_be_colored_input in df.columns:
                print(f'\nRecognizing ["{var_to_be_colored_input}"] as the variable to be colored.')
                var_to_be_colored = var_to_be_colored_input
            elif not var_to_be_colored_input:
                print('No coloring variable. Default color for plots will be navy blue.')
                break
            else:
                print('Error: Variable not found in system.')

        # Color coded graphing (if applicable)
        tuple_of_default_colors = ('black', 'red', 'blue', 'green', 'yellow', 'brown', 'purple')
        if var_to_be_colored and (len(df[var_to_be_colored].unique()) < len(tuple_of_default_colors)):
            colors = dict(zip(df[var_to_be_colored].unique(), tuple_of_default_colors))
            print(f'Color legend: {colors}')
            coloring = df[var_to_be_colored].apply(lambda x: colors[x])
            transparency_alpha = 0.5
        else:
            coloring = 'navy'
            if len(df) > 30:
                transparency_alpha = 0.3
            else:
                transparency_alpha = 0.5
        print('\nDisplaying graphs...')

        # Creating and plotting the scatter_matrix
        pd.plotting.scatter_matrix(df, alpha=transparency_alpha, figsize=(8, 8), diagonal='hist',
                                   c='None', s=20, edgecolors=coloring,
                                   hist_kwds={'color': 'gray', 'ec': 'black', 'bins': 20, 'density': True})

        # Display settings and show plot
        plt.tight_layout()
        plt.show()


def proc_convert_to_input(df, dependent):
    # I. Removing remaining object columns (IF YOU DON'T WANT THIS, PROCESS YOUR OBJECTS !!)
    past_num_of_vars = len(df.columns)
    dependent_type = df[dependent].dtype
    dependent_mapping = ''
    if dependent_type == 'object':
        df[dependent] = df[dependent].astype('category')
        dependent_type = 'category'
    if dependent_type == 'category':
        dependent_mapping = dict(enumerate(df[dependent].cat.categories))
        df[dependent] = df[dependent].astype('category').cat.codes

    df = df.select_dtypes(exclude='object')
    current_num_of_vars = len(df.columns)

    # II. Drop all rows with any NaN values (IF YOU DON'T WANT THIS, PROCESS YOUR NAN !!)
    num_of_nan_rows = df.isna().sum().sum()
    df.dropna(how='any', axis=0, inplace=True)
    current_num_of_rows = len(df)

    # III. Printing post-auto-truncating df characteristics
    print(''.join([f'\n {"-" * 80}\n', '%-81s' % f'| Deleted {past_num_of_vars - current_num_of_vars} '
                                                 f'qualitative vars, and dropped {num_of_nan_rows} NaN values', '|\n',
                   '%-81s' % f'| Remaining variables: {current_num_of_vars}', '|\n',
                   '%-81s' % f'| Remaining observations: {current_num_of_rows}', f'|\n {"-" * 80}']))
    if current_num_of_vars > (current_num_of_rows / 10):
        print('**** WARNING: > 1 vars for every 10 observations ****')

    return df, dependent_mapping


# III. Data processing
def format_string_length(string, length, string_type='s'):
    string = ''.join(['%', str(length), string_type]) % string
    return string


def df_set_row_header(df, header_row):
    df.columns = df.iloc[header_row]
    df.drop(header_row, axis=0, inplace=True)


def vars_unpivot(df, vars_to_unpivot, unpivot_column_name='Unpivot', unpivot_value_column_name='Value'):
    df_unpivot = pd.melt(df,
                         id_vars=tuple(df.drop(vars_to_unpivot, axis=1).columns),
                         var_name=unpivot_column_name,
                         value_name=unpivot_value_column_name)
    return df_unpivot


def var_pivot(df, name_of_column_to_be_pivot):
    df_pivot = pd.pivot(df, None, name_of_column_to_be_pivot)
    return df_pivot


def var_one_hot(df, column_name):
    df[column_name] = df[column_name].astype('category')
    df = pd.concat([df, pd.get_dummies(df[column_name], prefix=column_name).astype(int)], axis=1)
    df = df.drop(column_name, axis=1)

    return df


def var_to_numerical_cat(df, column_name):
    df[column_name] = df[column_name].astype('category').cat.codes
    return df


def df_reset_index(df, reset_index_check):
    if reset_index_check:
        df.reset_index(drop=True, inplace=True)


def generator_from_list(list_of_items_to_generate):
    for item_to_generate in list_of_items_to_generate:
        yield item_to_generate


def make_title(text_in_middle):
    return f"{'<' * 80} {text_in_middle} {'>' * 100}"


# IV. MODELING
def add_interactions(x, list_of_interactions):
    for interaction in list_of_interactions:
        try:
            splitted_interact = interaction.split(':')
            x[interaction] = x[splitted_interact[0]] * x[splitted_interact[1]]
        except KeyError:
            print('One of the interactions is invalid. The custom model will be terminated.')
            return True
    return False


def add_all_interactions(x):
    original_x_len = len(x.columns)
    for var in range(0, original_x_len):
        for next_var in range(var + 1, original_x_len):
            x[f'{x.columns[var]}:{x.columns[next_var]}'] = \
                x[x.columns[var]] * x[x.columns[next_var]]


def add_poly(x, list_of_poly_vars, list_of_degree):
    if list_of_poly_vars and list_of_degree:
        for poly_var_index in range(0, len(list_of_poly_vars)):
            current_poly_var_name = list_of_poly_vars[poly_var_index]
            for degree in range(2, list_of_degree[poly_var_index] + 1):
                try:
                    x[f"{current_poly_var_name}^{degree}"] \
                        = x[current_poly_var_name] ** degree
                except KeyError:
                    print('One of the polies is invalid. The custom model will be terminated.')
                    return True
    return False


def add_all_poly(x, degree):
    original_x_len = len(x.columns)
    for var_index in range(0, original_x_len):
        for degree_level in range(2, degree + 1):
            x[f"{x.columns[var_index]}^{degree_level}"] \
                = x[x.columns[var_index]] ** degree_level


def drop_vars_for_modeling(x, list_of_vars_to_drop):
    if list_of_vars_to_drop:
        try:
            x.drop(list_of_vars_to_drop, axis=1, inplace=True)
        except KeyError:
            print('One of the vars to drop is invalid. The custom model will be terminated.')
            return True
    return False


def custom_x_transform(x, list_of_drop, list_of_poly, list_of_degrees, list_of_interactions,
                       all_interactions, all_poly, all_poly_degree):

    check_error = False
    transformations_done = []

    if list_of_drop or (list_of_poly and list_of_degrees) or list_of_interactions or all_interactions or \
            (all_poly and all_poly_degree):

        poly_error = bool
        interaction_error = bool
        dropping_error = bool

        # Add polynomials
        if list_of_poly and list_of_degrees:
            poly_error = add_poly(x, list_of_poly, list_of_degrees)
            transformations_done.append('add poly')

        # Add ALL possible polinomials (bringing the equation to quadratic form, only works if polies are not specified)
        if not poly_error and all_poly and all_poly_degree and not list_of_poly and not list_of_degrees:
            add_all_poly(x, all_poly_degree)
            transformations_done.append('add all poly')

        # Add interactions
        if not poly_error and list_of_interactions:
            interaction_error = add_interactions(x, list_of_interactions)
            transformations_done.append('add interactions')

        # Add ALL possible interactions (only if LIST_OF_INTERACTIONS is not specified)
        if not interaction_error and not poly_error and all_interactions and not list_of_interactions:
            add_all_interactions(x)
            transformations_done.append('add all interactions')

        # Dropping specified variables
        if not interaction_error and not poly_error and list_of_drop:
            dropping_error = drop_vars_for_modeling(x, list_of_drop)
            transformations_done.append('dropped custom variables')

        check_error = interaction_error and poly_error and dropping_error or False

    return check_error, transformations_done


# LINEAR REGRESSION
def linear_regression_main(
        x, y, l1_or_l2=1, penalty=0, cross_validation=1,
        individual_reg=True, quadratic_test_linear=True
):
    x_with_const = sm.add_constant(x)
    cross_validation_check = False
    if cross_validation > 1:
        cross_validation_check = True

    # Linear regression for each individual variable (included interactions)
    if individual_reg:
        individual_linear_reports = linear_regression_simple_individuals(x=x, y=y)
    else:
        individual_linear_reports = '(No individual reports)\n\n'

    # Simple multiple linear regression including all variables (included interactions)
    linear_model = sm.OLS(y, x_with_const).fit()  # Regular multiple linear regression
    pred_y = linear_model.predict(x_with_const)
    if not cross_validation_check:
        cross_validated_mse = smte.mse(pred_y, y)
    else:
        cross_validated_mse = linear_regression_cv(x_with_const=x_with_const, y=y,
                                                   model='normal_fit', l1_or_l2=l1_or_l2, penalty=penalty,
                                                   cross_validation=cross_validation)

    for individual_linear_report in individual_linear_reports:
        print(individual_linear_report)
    print(linear_model.summary())
    print()
    print(f'MSE mean cross-validated [{cross_validation_check}]: {str(round(cross_validated_mse, 4))}')
    print(f'RMSE cross-validated [{cross_validation_check}]: {str(round(cross_validated_mse * 0.05, 4))}')
    print(f'MSE train: {str(round(smte.mse(pred_y, y), 4))}')
    print(f'MAE train: {str(round(smte.meanabs(pred_y, y), 4))}')
    print(f'AIC: {linear_model.aic}; BIC: {linear_model.bic}')

    # Regularized multiple linear regression including all variables (included interactions)
    print(f'\n{"-" * 80}\n[REGULARIZED REGRESSION]\n')
    lasso_model = sm.OLS(y, x_with_const).fit_regularized(L1_wt=l1_or_l2, alpha=penalty)
    if not cross_validation_check:
        cross_validated_mse = smte.mse(lasso_model.predict(x_with_const), y)
    else:
        cross_validated_mse = linear_regression_cv(x_with_const=x_with_const, y=y,
                                                   model='regularized', l1_or_l2=l1_or_l2, penalty=penalty,
                                                   cross_validation=cross_validation)
    print(lasso_model.params)
    print()
    print(f'MSE mean cross-validated [{cross_validation_check}]: {str(round(cross_validated_mse, 4))}')
    print(f'RMSE cross-validated [{cross_validation_check}]: {str(round(cross_validated_mse * 0.05, 4))}')
    print(f'MSE single test: {str(round(smte.mse(lasso_model.predict(x_with_const), y), 4))}')
    print(f'MAE single test: {str(round(smte.meanabs(lasso_model.predict(x_with_const), y), 4))}')
    print(f'AIC: {linear_model.aic}; BIC: {linear_model.bic}')

    if quadratic_test_linear:
        print(f'\n{"-" * 80}\n[QUADRATIC TEST]\n')
        linear_for_quadratic_dict, linear_for_quadratic_dict_regularized = \
            linear_regression_quadratic_test(x=x, y=y, l1_or_l2=LASSO_OR_RIDGE, penalty=PENALTY,
                                             cross_validation=K_FOLD_CV,
                                             quadratic_test_to_degree=QUADRATIC_TEST_TO_DEGREE)
        print(linear_for_quadratic_dict)
        print(linear_for_quadratic_dict_regularized)
        figs, axs = plt.subplots(2)

        x_plot, y_plot = zip(*sorted(linear_for_quadratic_dict.items()))
        axs[0].plot(x_plot, y_plot)
        axs[0].title.set_text('Linear regression MSE (y) by degrees (x)')

        x_plot, y_plot = zip(*sorted(linear_for_quadratic_dict_regularized.items()))
        axs[1].plot(x_plot, y_plot)
        axs[1].title.set_text('Linear regression regularized MSE (y) by degrees (x)')
        plt.tight_layout()
        plt.show()

    # ------------------------ LINEAR REGRESSION: SOLVING TYPICAL LINEAR PROBLEMS -------------------------------- #
    print(f'\n{"-" * 80}\n[TEST FOR POLY TREND, OUTLIERS, AND HIGH LEVERAGE OBSERVATIONS]\n')
    linear_regression_simple_poly_outlier_leverage_plot(linear_model)


def linear_regression_simple_individuals(x, y):
    x_with_const = sm.add_constant(x)
    report = []
    for column in x.columns:
        linear_model = sm.OLS(y, x_with_const[['const', column]]).fit()
        current_var_p_value = linear_model.pvalues[1]
        report.append(f'Param regressed: {column}\n'
                      f'Coefficient: {round(linear_model.params[1], 8)}\n'
                      f'Significance: {current_var_p_value < ALPHA} ({round(current_var_p_value, 8)})\n')
    return report


def linear_regression_cv(x_with_const, y, model, l1_or_l2, penalty, cross_validation):
    loop_cross_val = cross_validation
    mse_mean = 0

    if model == 'normal_fit' and loop_cross_val > 1:
        fold_x = list(np.array_split(x_with_const, cross_validation))
        fold_y = list(np.array_split(y, cross_validation))
        while loop_cross_val > 1:
            # Loop to get mse
            train_x = fold_x[:]
            train_y = fold_y[:]
            test_x = train_x.pop(loop_cross_val-1)
            test_y = train_y.pop(loop_cross_val-1)
            mse_mean += smte.mse(sm.OLS(pd.concat(train_y, axis=0), pd.concat(train_x, axis=0))
                                 .fit().predict(test_x), test_y)
            loop_cross_val -= 1
        mse_mean = mse_mean / cross_validation

    elif model == 'regularized' and loop_cross_val > 1:
        fold_x = np.array_split(x_with_const, cross_validation)
        fold_y = np.array_split(y, cross_validation)
        while loop_cross_val > 1:
            # Loop to get mse
            train_x = fold_x[:]
            train_y = fold_y[:]
            test_x = train_x.pop(loop_cross_val - 1)
            test_y = train_y.pop(loop_cross_val - 1)
            mse_mean += smte.mse(sm.OLS(pd.concat(train_y, axis=0), pd.concat(train_x, axis=0))
                                 .fit_regularized(L1_wt=l1_or_l2, alpha=penalty).predict(test_x), test_y)
            loop_cross_val -= 1
        mse_mean = mse_mean / cross_validation

    return mse_mean


def linear_regression_quadratic_test(x, y, l1_or_l2, penalty, cross_validation, quadratic_test_to_degree):
    mse_mean_list = []
    mse_mean_list_regularized = []
    list_of_test_degree = list(range(1, quadratic_test_to_degree+1))
    for degree in list_of_test_degree:
        quadratic_test_x = x.copy()
        add_all_poly(x=quadratic_test_x, degree=degree)
        x_with_const = sm.add_constant(quadratic_test_x)
        mse_mean = linear_regression_cv(x_with_const=x_with_const, y=y, model='normal_fit',
                                        l1_or_l2=l1_or_l2, penalty=penalty, cross_validation=cross_validation)
        mse_mean_regularized = linear_regression_cv(x_with_const=x_with_const, y=y, model='regularized',
                                                    l1_or_l2=l1_or_l2, penalty=penalty,
                                                    cross_validation=cross_validation)
        mse_mean_list.append(mse_mean)
        mse_mean_list_regularized.append(mse_mean_regularized)

    linear_for_quadratic_dict = dict(zip(list_of_test_degree, mse_mean_list))
    linear_for_quadratic_dict_regularized = dict(zip(list_of_test_degree, mse_mean_list_regularized))

    return linear_for_quadratic_dict, linear_for_quadratic_dict_regularized


def linear_regression_simple_poly_outlier_leverage_plot(linear_model):
    fig, axs = plt.subplots(2)

    linear_studentized_residual = tuple(linear_model.get_influence().resid_studentized_external)
    linear_fitted_values = tuple(linear_model.fittedvalues)
    linear_leverage = tuple(linear_model.get_influence().hat_matrix_diag)
    linear_studentized_residual_color = []

    for index in range(len(linear_studentized_residual)):
        if linear_studentized_residual[index] > 3 or linear_studentized_residual[index] < -3:
            linear_studentized_residual_color.append('red')
            axs[0].annotate(index, (linear_fitted_values[index], linear_studentized_residual[index]))
            axs[1].annotate(index, (linear_leverage[index], linear_studentized_residual[index]))
        else:
            linear_studentized_residual_color.append('navy')

    # Finding possible poly trend and possible outliers (in red, studentized residual > 3 or < -3)
    axs[0].scatter(linear_fitted_values, linear_studentized_residual, c='None',
                   edgecolors=linear_studentized_residual_color, s=20, alpha=0.3)
    axs[0].title.set_text(f'LINEAR: Studentized residual (y) versus predicted value (x)')

    # Finding high leverage points (red points with significantly higher leverage than rest of data)
    axs[1].scatter(linear_leverage, linear_studentized_residual, c='None',
                   edgecolors=linear_studentized_residual_color, s=20, alpha=0.3)
    axs[1].title.set_text("LINEAR: Studentized residual (y) versus leverage (x)")

    plt.tight_layout()
    plt.show()


# CLASSIFICATION
def acc_fun(target_true, target_fit):
    target_fit = np.round(target_fit)
    target_fit.astype('int')
    return metrics.accuracy_score(target_true, target_fit)


def classification_convert_probability_weights(y, weights_list):
    labels_list = y.unique()
    if weights_list and len(weights_list) == len(labels_list):
        class_weight = dict(zip(labels_list, weights_list))
    else:
        class_weight = 'balanced'

    return class_weight


def classification_simple_logit_ovr(x, y):
    features = sm.add_constant(x)
    all_features = features.columns[1:]
    all_classes = y.unique()

    list_of_report = []
    list_of_train_error_rate = []

    for label in all_classes:
        target = np.where(y == label, 1, 0)

        try:
            logit = sm.Logit(target, features).fit(disp=False)
        except sm.tools.sm_exceptions.PerfectSeparationError:
            logit = sm.Logit(target, features).fit(method='bfgs', disp=False)

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


def classification_mn_logit(x, y, model_if_newton_fails='bfgs'):
    features = sm.add_constant(x)
    all_classes = y.unique()
    all_features = features.columns[1:]

    try:
        mnlogit = sm.MNLogit(y, features).fit(disp=False)
    except sm.tools.sm_exceptions.PerfectSeparationError:
        mnlogit = sm.MNLogit(y, features).fit(method=model_if_newton_fails, disp=False)

    train_pred_target = list(pd.DataFrame(mnlogit.predict(features)).idxmax(axis=1))
    for index in range(len(train_pred_target)):
        train_pred_target[index] = all_classes[train_pred_target[index]]
    train_error_rate = round(metrics.accuracy_score(y, train_pred_target), 4)

    data = dict(zip([f'mff_{item}' for item in all_classes],
                    np.round(np.transpose(mnlogit.get_margeff().margeff), 4)))
    data2 = dict(zip([f'stdErr_{item}' for item in all_classes],
                     np.round(np.transpose(mnlogit.get_margeff().margeff_se), 4)))
    data3 = dict(zip([f'pvalues_{item}' for item in all_classes],
                     np.round(np.transpose(mnlogit.get_margeff().pvalues), 4)))

    data.update(data2)
    data.update(data3)
    report = pd.DataFrame(data=data, index=all_features)

    return report, train_error_rate, train_pred_target


def classification_mn_logit_regularized(x, y, penalty):
    """Note: function will perform l1 regularization if penalty is set to > 0"""
    features = sm.add_constant(x)
    all_classes = y.unique()
    all_features = features.columns[1:]

    mnlogit = sm.MNLogit(y, features).fit_regularized(disp=False, alpha=penalty)

    train_pred_target = list(pd.DataFrame(mnlogit.predict(features)).idxmax(axis=1))
    for index in range(len(train_pred_target)):
        train_pred_target[index] = all_classes[train_pred_target[index]]
    train_error_rate = round(metrics.accuracy_score(y, train_pred_target), 4)

    data = dict(zip([f'mff_{item}' for item in all_classes],
                    np.round(np.transpose(mnlogit.get_margeff().margeff), 4)))
    data2 = dict(zip([f'stdErr_{item}' for item in all_classes],
                     np.round(np.transpose(mnlogit.get_margeff().margeff_se), 4)))
    data3 = dict(zip([f'pvalues_{item}' for item in all_classes],
                     np.round(np.transpose(mnlogit.get_margeff().pvalues), 4)))

    data.update(data2)
    data.update(data3)
    report = pd.DataFrame(data=data, index=all_features)

    return report, train_error_rate, train_pred_target


def classification_cv_poly_model_test(x, y, model_list, list_of_model_name,
                                      cross_validation, quadratic_test_to_degree):
    accuracy_scoring = metrics.make_scorer(acc_fun)

    confusion_matrix_list = []
    poly_error_list = []

    if quadratic_test_to_degree > 1:
        tuple_of_test_degree = tuple(range(1, quadratic_test_to_degree + 1))
        quadratic_test_y = y.copy()
        quadratic_test_y = quadratic_test_y.astype('category').cat.codes

        for degree in tuple_of_test_degree:
            cv_error_list = []
            quadratic_test_x = x.copy()
            add_all_poly(x=quadratic_test_x, degree=degree)

            for test_model in model_list:
                if degree == 1:
                    fitted_model = test_model.fit(X=quadratic_test_x, y=quadratic_test_y)
                    confusion_matrix_list.append(classification_confusion_matrix(
                        labels=quadratic_test_y,
                        pred_labels=np.round(fitted_model.predict(quadratic_test_x), 0),
                        all_names_not_categorized=y.unique().astype('object')))
                cv_error_list.append(np.round(np.mean(cross_val_score(estimator=test_model, X=quadratic_test_x,
                                                                      y=quadratic_test_y, cv=cross_validation,
                                                                      scoring=accuracy_scoring)), 4))
            poly_error_list.append(dict(zip(list_of_model_name, cv_error_list)))

        report = pd.DataFrame(dict(zip(tuple_of_test_degree, poly_error_list)))
        confusion_matrix_dict = dict(zip(list_of_model_name, confusion_matrix_list))

        return report, confusion_matrix_dict


def classification_main(x, y, multinomial_model, penalty, cross_validation, list_of_weights,
                        quadratic_test_to_degree, individual_reg):
    # Weight converting
    class_weights = classification_convert_probability_weights(y=y, weights_list=list_of_weights)

    # Model initiations
    model_linear = LinearRegression()
    model_one_vs_rest = LogisticRegression(multi_class='ovr',
                                           class_weight=class_weights)
    model_multi = LogisticRegression(multi_class='multinomial',
                                     solver=multinomial_model,
                                     class_weight=class_weights)
    model_ordinal = LogisticAT(alpha=penalty)  # alpha = 0 means no regularization

    model_list = [model_linear, model_one_vs_rest, model_multi, model_ordinal]
    list_of_model_name = ['Linear Least Square', 'Logistic OVR', 'Logistic Multinomial', 'Logistic Ordinal']

    one_dict_of_reports = None
    one_dict_of_train_error_rates = None

    # Simple logistic regression for each class (marginal effects, std err, and p-values included)
    if individual_reg:
        one_dict_of_reports, one_dict_of_train_error_rates = classification_simple_logit_ovr(x, y)

    # Multinomial logistic regression (marginal effects, std err, and p-values included)
    two_report, two_train_error_rate, two_train_pred_target = classification_mn_logit(
        x=x, y=y, model_if_newton_fails='lbfgs')

    three_report, three_train_error_rate, three_train_pred_target = classification_mn_logit_regularized(
        x=x, y=y, penalty=penalty)

    # Quadratic test for each model
    fourth_report, confusion_matrix_dict = classification_cv_poly_model_test(
        x=x, y=y, model_list=model_list, list_of_model_name=list_of_model_name,
        cross_validation=cross_validation, quadratic_test_to_degree=quadratic_test_to_degree)

    # -------------------------------------REPORT------------------------------------
    if one_dict_of_reports:
        print(f"\n{make_title('SIMPLE LOGISTIC REGRESSION WITH EACH CLASS (BINARY)')}")

        for key, value in one_dict_of_reports.items():
            print(f'[{key}]\n{value}\n\n')

        print('[ERROR RATE RESULTS]')

        for key, value in one_dict_of_train_error_rates.items():
            print(f'{key}: {1 - value}')

    print(f"\n{make_title('MULTINOMIAL LOGISTIC REGRESSION WITH ALL CLASSES')}")
    print(f'{two_report}\n\nTrain Error rate: {1 - two_train_error_rate}')

    print(f"\n{make_title('MULTINOMIAL LOGISTIC REGRESSION REGULARIZED')}")
    print(f'{three_report}\n\nTrain Error rate: {1 - three_train_error_rate}')

    print(f"\n{make_title('CLASSIFICATION MODELS AND POLY TEST')}")
    print(fourth_report)

    print('\n[CONFUSION MATRICES]')
    print(class_weights)
    print()

    for key, value in confusion_matrix_dict.items():
        print(f'[{key}]\n{value}\n\n')
    fourth_report.T.plot()
    plt.show()


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


def main(current_file_link):
    df, not_csv_alerts = proc_importing(current_file_link=current_file_link, index_of_sheet=INDEX_SHEET,
                                        frac=FRAC, variable_types=VARIABLE_TYPES)
    if df.empty:
        print('Empty dataframe.')
        return

    print('\nFinished importing data.\n')

    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    proc_display_raw(df)
    df, dfs_to_export = proc_processing(df)

    link_to_folder = current_file_link.split('\\')
    name_of_current_file = link_to_folder.pop(-1)
    name_of_current_file_splitted = name_of_current_file.split('.')
    name_of_current_file = name_of_current_file_splitted[0]
    current_file_type = name_of_current_file_splitted[1]

    # Printing overview of all files in system
    print(f'\n\n{current_file_link}')
    print(f" {'-' * 99}\n"
          f"{'%-100s' % f'|  Current file: {name_of_current_file}'}|\n"
          f"{'%-100s' % f'|  File type: {current_file_type}'}|\n"
          f"{not_csv_alerts}"
          f" {'-' * 99}\n")

    # Check for export call
    proc_exporting(df=df, dfs_to_export=dfs_to_export,
                   link_to_folder=link_to_folder, name_of_current_file=name_of_current_file)

    # *********************** FURTHER DATA TRUNCATING + PRELIMINARY GRAPHING FOR ANALYSIS *************************** #
    # Specifying the dependent
    dependent = DEPENDENT
    while not DEPENDENT:
        dependent_input = input('What shall be the dependent variable?: ')
        if dependent_input in df.columns:
            dependent = dependent_input
            break
        print('Error: Variable not found in system.')

    df, dependent_mapping = proc_convert_to_input(df, dependent)

    # IV. Scatter matrix graphing of df
    print(df)
    print(f'\nThe dependent has been set as ["{dependent}"] for modeling purposes.')
    if dependent_mapping:
        print(dependent_mapping)
    proc_scatter_matrix(df)

    # ********************************************** MODEL BUILDING ************************************************* #
    y = df[dependent]
    x = df.drop(dependent, axis=1)

    # --------------------------------------------- LINEAR REGRESSION -------------------------------------------------#
    #  Questions to ask:                                                                                               #
    #   1. Is there a relationship between the dependent and any of the variables? (use F-stat in multiple linear reg) #
    #   2. How strong is the relationship? (use adjusted R-squared in multiple linear reg)                             #
    #   3. Which variables should be included? (use each var's t-stat and p-value in multiple linear reg)              #
    #   4. How large does each variable affect the dependent? (use each var's confidence interval of coefficient in    #
    #                                                               multiple linear reg, as well as in individual reg) #
    #   5. How accurate can we predict future dependent values? (use prediction interval to predict individual response#
    #               - wider to account for irreducible errors - or confidence interval to predict on average response) #
    #   6. Is the relationship linear? (use studentized residual spot trend & outliers)                                #
    #                                                                                                                  #
    # --------------------------------------------- LINEAR REGRESSION -------------------------------------------------#
    if LINEAR:
        x_for_linear = x.copy()
        check_error, transformation_done = custom_x_transform(
            x_for_linear, LIST_TO_DROP, LIST_OF_POLY_VARS, LIST_OF_DEGREE,
            LIST_OF_INTERACTIONS, ALL_INTERACTIONS, ALL_POLY, ALL_POLY_DEGREE
        )

        if check_error:
            print('ERRORS DURING TRANSFORMATION OF X')
            return

        print(make_title('LINEAR REGRESSIONS'))
        if transformation_done:
            print(transformation_done)

        linear_regression_main(x=x_for_linear, y=y, individual_reg=True,
                               quadratic_test_linear=QUADRATIC_TEST_LINEAR,
                               l1_or_l2=LASSO_OR_RIDGE, penalty=PENALTY,
                               cross_validation=K_FOLD_CV)

    # ---------------------------------------------- CLASSIFICATION -------------------------------------------------- #
    #  Questions to ask:                                                                                               #
    #   1. Is there a relationship between the dependent and any of the variables? (use F-stat in multiple linear reg) #
    #   2. How strong is the relationship? (use adjusted R-squared in multiple linear reg)                             #
    #   3. Which variables should be included? (use each var's t-stat and p-value in multiple linear reg)              #
    #   4. How large does each variable affect the dependent? (use each var's confidence interval of coefficient in    #
    #                                                               multiple linear reg, as well as in individual reg) #
    #   5. How accurate can we predict future dependent values? (use prediction interval to predict individual response#
    #               - wider to account for irreducible errors - or confidence interval to predict on average response) #
    #   6. Is the relationship linear? (use studentized residual to spot trend & outliers)                             #
    #                                                                                                                  #
    # ---------------------------------------------- CLASSIFICATION -------------------------------------------------- #
    if CLASSIFICATION:
        x_for_classification = x.copy()

        # -------------------------- LOGISTIC REGRESSION -------------------------------- #
        classification_main(x=x_for_classification, y=y, multinomial_model='saga',
                            penalty=PENALTY, cross_validation=K_FOLD_CV, list_of_weights=LIST_OF_WEIGHTS,
                            quadratic_test_to_degree=QUADRATIC_TEST_TO_DEGREE, individual_reg=True)

        # K-nearest neighbors
        print(f"\n{make_title('K-NEAREST NEIGHBORS')}")
        knn_classification = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS).fit(x_for_classification, y)
        print(knn_classification)

        # ------------------------------ LOGISTIC REGRESSION WITH CUSTOMIZED VARS ------------------------------------ #
        early_termination_check, transformation_done = custom_x_transform(
            x_for_classification, LIST_TO_DROP, LIST_OF_POLY_VARS, LIST_OF_DEGREE,
            LIST_OF_INTERACTIONS, ALL_INTERACTIONS, ALL_POLY, ALL_POLY_DEGREE
        )

        if not early_termination_check:
            print(f"\n{make_title('CLASSIFICATION WITH CUSTOMIZED VARS')}")

    print('\nEnd of program.')


# 1. ********************************************* MAIN ******************************************************** #
if __name__ == '__main__':
    main(current_file_link=CURRENT_FILE_LINK)
