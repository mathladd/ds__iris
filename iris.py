import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask_ml.preprocessing import DummyEncoder, StandardScaler

import pandas as pd
import numpy as np

import warnings
from joblib import parallel_backend as joblib_parallel_backend
import os

# ************************************************** SETUPS ******************************************************* #
pd.set_option('display.expand_frame_repr', False)
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=FutureWarning)


# 1. ********************************************* GLOBAL KEYS **************************************************** #
LINK_TO_FILE = r'C:\Users\Alien\Downloads\datasets\dataset_titanic\train.csv'
# C:\Users\CB495EF\Downloads\self projects\data\dataset_titanic\train.csv
# C:\Users\CB495EF\Downloads\self projects\data\dataset_momo\Mercury vol2 Q3.csv
TARGET = 'Survived'

ALL_FEATURE_TYPES: dict = {'PassengerId': 'int64'}
USED_FEATURES: list = []
ASSUME_MISSING_CHECK: bool = False
PROC_ANALYSIS_CHECK: bool = True
CONVERT_ALL_OBJECTS_TO_CAT_CHECK: bool = False
PRE_PROCESSING_TO_PARQUET_CHECK: bool = False
POST_PROCESSING_TO_PARQUET_CHECK: bool = False


"""Distributed settings"""
ADDRESS: str = 'local'


"""Modeling"""
GET_DUMMY_CHECK: bool = True
STANDARDIZE_CHECK: bool = True


# 2. ********************************************* ALL FUNCTIONS **************************************************** #
def main():
    # Variable declarations
    # ------------------------------------------------------------------------------------------------------------
    link_to_file = LINK_TO_FILE
    all_feature_types = ALL_FEATURE_TYPES
    used_features = USED_FEATURES
    assume_missing_check = ASSUME_MISSING_CHECK
    proc_analysis_check = PROC_ANALYSIS_CHECK
    convert_all_objects_to_cat_check = CONVERT_ALL_OBJECTS_TO_CAT_CHECK
    pre_processing_to_parquet_check = PRE_PROCESSING_TO_PARQUET_CHECK
    post_processing_to_parquet_check = POST_PROCESSING_TO_PARQUET_CHECK

    address = ADDRESS

    get_dummy_categorical = GET_DUMMY_CHECK
    standardize_numeric = STANDARDIZE_CHECK

    # Preliminary checks and settings
    # ------------------------------------------------------------------------------------------------------------
    link_to_folder = link_to_file.split('\\')
    file_name = link_to_folder.pop(-1)
    link_to_folder = '\\'.join(link_to_folder) + '\\'
    possible_link_to_parquet_file = f'{link_to_folder}{file_name.split(".")[0]}.parquet'
    check_existing_parquet = os.path.exists(possible_link_to_parquet_file)
    print(f'\nReading: {link_to_file}')

    if check_existing_parquet:
        link_to_file = possible_link_to_parquet_file
        file_name = link_to_file.split('\\').pop(-1)
        print(f'Parquet at: {possible_link_to_parquet_file}')

    df_test = proc_import(link_to_file=link_to_file, file_name=file_name, all_feature_types=all_feature_types,
                          assume_missing=assume_missing_check, used_features=used_features)
    if not df_test.columns.tolist():
        print('Empty DataFrame')
        return
    all_features = df_test.columns.tolist()
    number_of_features = len(all_features)
    print(f'Number of features: {number_of_features}')

    # Data importation
    # ------------------------------------------------------------------------------------------------------------
    if address == 'local':
        local_cluster = LocalCluster(processes=True, threads_per_worker=1)
        address = local_cluster
    client = Client(address=address, set_as_default=True)

    if not np.array_equal(all_feature_types.keys(), all_features):
        all_feature_types_suggested_ = dict(zip(all_features, df_test.dtypes.astype(str).tolist()))
        for key, value in all_feature_types.items():
            if key in all_feature_types:
                all_feature_types_suggested_[key] = value
        all_feature_types = all_feature_types_suggested_

    if proc_analysis_check:
        analysis_report, all_feature_types_suggested, number_of_observations, numeric_features, \
            category_features, object_features, date_features = proc_analysis(
                all_features=all_features, link_to_file=link_to_file, file_name=file_name,
                all_feature_types=all_feature_types, convert_all_objects_to_cat=convert_all_objects_to_cat_check,
                assume_missing=assume_missing_check, client=client)
    else:
        analysis_report = ''
        all_feature_types_suggested = all_feature_types
        number_of_observations = -1
        numeric_features, category_features, object_features, date_features = [], [], [], []
        
        for key, value in all_feature_types_suggested.items():
            if 'int' in value or 'float' in value:
                numeric_features.append(key)
            elif 'category' in value:
                category_features.append(key)
            elif 'object' in value and convert_all_objects_to_cat_check:
                category_features.append(key)
            elif 'object' in value and not convert_all_objects_to_cat_check:
                object_features.append(key)
            elif 'datetime' in value:
                date_features.append(key)

    df = proc_import(link_to_file=link_to_file, file_name=file_name, all_feature_types=all_feature_types_suggested,
                     assume_missing=assume_missing_check, used_features=used_features)
    print(f'Returning data broken into {df.npartitions} partitions')

    # Raw data display
    # ------------------------------------------------------------------------------------------------------------
    print('-' * 400)
    print('[RAW DATA DISPLAY: HEAD AND TAIL]')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.concat([df.head(5), df.tail(5)], axis=0))
    print(f'\nNumber of features: {number_of_features}')
    print(f'Number of observations: {number_of_observations}')
    print(f'Input dtypes:\n{all_feature_types}')
    print(f'Suggested dtypes:\n{all_feature_types_suggested}')
    print(f'\nAll numeric features:\t\t{numeric_features}')
    print(f'All categorical features:\t{category_features}')
    print(f'All object features:\t\t{object_features}')
    print(f'All datetime features:\t\t{date_features}')

    # Analysis report
    # ------------------------------------------------------------------------------------------------------------
    if proc_analysis_check:
        print('-' * 400)
        print('[PROC ANALYSIS]')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 100):
            print(analysis_report)
            print('-' * 400)
            print('[CORRELATION MATRIX]')
            print(df.corr().compute())

    # Model input processing
    # ------------------------------------------------------------------------------------------------------------
    print('-' * 400)
    df, export_list = proc_process(df=df)
    if pre_processing_to_parquet_check:
        df.to_parquet(possible_link_to_parquet_file)
        print('Export main file to parquet completed for subsequent importations.')
    
    if export_list:
        for export_index in range(len(export_list)):
            export_list[export_index].to_csv(f'{link_to_folder}{file_name.split(".")[0]} ({export_index}).csv')
            
    df = df.drop(object_features, axis=1)
    print('Dropped all object features.')
    df = df.drop(date_features, axis=1)
    print('Dropped all date features.')
    df = df.dropna(how='any')
    print('Dropped all null observations.')
    df = df.reset_index(drop=True)

    if standardize_numeric:
        scale = StandardScaler()
        scale.fit(df[numeric_features])
        df[numeric_features] = scale.transform(df[numeric_features])
        print('Standardized all numeric features.')

    if get_dummy_categorical:
        df, base_categories = transform_get_dummies(df, category_features)
        print(f'Got dummies for categorical. Base case: {base_categories}')

    # Final look at df blueprint
    # ------------------------------------------------------------------------------------------------------------
    print('-' * 400)
    print(df)

    # Post-processing to parquet
    # ------------------------------------------------------------------------------------------------------------
    print('-' * 400)
    if post_processing_to_parquet_check:
        df.to_parquet(f'{link_to_folder}{file_name.split(".")[0]}_processed.parquet')
        print('Export post-processing data completed.')

    # Inititate regression
    # ------------------------------------------------------------------------------------------------------------
    with joblib_parallel_backend('dask'):
        regression()

    # End of main
    # ------------------------------------------------------------------------------------------------------------
    address.close()
    client.close()


def proc_process(df):
    """Pre-processing"""
    # --------------------------------- Code starts here ---------------------------------------- #
    #

    #
    # --------------------------------- Code ends here ------------------------------------------ #
    """Calculations and printing out results"""
    # --------------------------------- Code starts here ---------------------------------------- #
    #

    #
    # --------------------------------- Code ends here ------------------------------------------ #
    export_list = []
    return df, export_list


def proc_import(link_to_file: str, file_name: str, all_feature_types: dict, assume_missing: bool, used_features: list):
    # Check if all_feature_types are specified
    if not all_feature_types:
        all_feature_types = None

    # Check if used_cols_indices are specified
    if not used_features:
        used_features = None

    # If csv file type
    if 'csv' in file_name:
        df = dd.read_csv(urlpath=link_to_file, dtype=all_feature_types, assume_missing=assume_missing,
                         usecols=used_features, skip_blank_lines=True, sep=',')
    elif 'parquet' in file_name:
        df = dd.read_parquet(path=link_to_file, columns=used_features)
    else:
        df = pd.DataFrame()

    # Return dask dataframe
    return df


def proc_analysis(all_features: list, link_to_file: str, file_name: str, all_feature_types: dict, assume_missing: bool,
                  convert_all_objects_to_cat: bool, client: Client):

    feature_name_, all_reports, all_dtypes, numerical_features, category_features, object_features, date_features = \
        [], [], [], [], [], [], []
    unique_display_width = 80
    unique_display_half = int(unique_display_width/2) - 5
    number_of_observations = 0
    half_number_of_observations = 0

    for feature_name in all_features:
        feature_name_.append(feature_name)
        current_feature = client.persist(proc_import(link_to_file=link_to_file, file_name=file_name,
                                                     all_feature_types=all_feature_types,
                                                     assume_missing=assume_missing, used_features=feature_name_))
        feature_name_.remove(feature_name)

        # Set number of observations ONCE ! ----------------------------------------------------------------
        if number_of_observations == 0:
            number_of_observations = len(current_feature)
            half_number_of_observations = int(number_of_observations/2)

        # Find dtype ---------------------------------------------------------------------------------------
        dtype = current_feature[feature_name].dtype.name

        # Count null values --------------------------------------------------------------------------------
        count_null = current_feature[feature_name].isnull().sum().compute()
        percent_null = ''
        problem = ''
        if count_null > 0:
            percent_null = str(round((count_null / number_of_observations) * 100, 1))
            problem = 'x'
        elif 'object' in dtype:
            problem = 'x'

        # Finding and counting uniques ---------------------------------------------------------------------
        uniques = current_feature[feature_name].unique()  # TODO may blow up memory
        nuniques = len(uniques)

        # Suggesting dtypes for features -------------------------------------------------------------------
        dtype_suggest = dtype
        sum_value, mean_value, std_value, min_value, twenty_five_perc_value, median_value, seventy_five_perc_value,\
            max_value = '', '', '', '', '', '', '', ''

        if 'int' in dtype or 'float' in dtype:
            numerical_features.append(feature_name)
            sum_value = current_feature[feature_name].sum()
            mean_value = current_feature[feature_name].mean()
            std_value = current_feature[feature_name].std()
            min_value = current_feature[feature_name].min()
            if count_null < number_of_observations:
                twenty_five_perc_value = current_feature[feature_name].quantile(0.25)
                median_value = current_feature[feature_name].quantile(0.5)
                seventy_five_perc_value = current_feature[feature_name].quantile(0.75)
            else:
                twenty_five_perc_value, median_value, seventy_five_perc_value = min_value, min_value, min_value
            max_value = current_feature[feature_name].max()
            sum_value, mean_value, std_value, min_value, twenty_five_perc_value, median_value, \
                seventy_five_perc_value, max_value = client.compute([sum_value, mean_value, std_value,
                                                                     min_value, twenty_five_perc_value, median_value,
                                                                     seventy_five_perc_value, max_value])
            sum_value = sum_value.result()
            mean_value = mean_value.result()
            std_value = std_value.result()
            min_value = min_value.result()
            twenty_five_perc_value = twenty_five_perc_value.result()
            median_value = median_value.result()
            seventy_five_perc_value = seventy_five_perc_value.result()
            max_value = max_value.result()
            if 'int' in dtype:
                if min_value > np.iinfo(np.int8).min and max_value < np.iinfo(np.int8).max:
                    dtype_suggest = 'int8'
                elif min_value > np.iinfo(np.int16).min and max_value < np.iinfo(np.int16).max:
                    dtype_suggest = 'int16'
                elif min_value > np.iinfo(np.int32).min and max_value < np.iinfo(np.int32).max:
                    dtype_suggest = 'in32'
                elif min_value > np.iinfo(np.int64).min and max_value < np.iinfo(np.int64).max:
                    dtype_suggest = 'int64'
            else:
                if min_value > np.finfo(np.float32).min and max_value < np.finfo(np.float32).max:
                    dtype_suggest = 'float32'
                else:
                    dtype_suggest = 'float64'

        # Suggest as category for object features with nuniques < 1/2(number_of_observations) --------------
        elif dtype == 'object':
            if convert_all_objects_to_cat or nuniques <= half_number_of_observations:
                dtype_suggest = 'category'
                category_features.append(feature_name)
            else:
                object_features.append(feature_name)
        elif dtype == 'datetime64[ns]':
            date_features.append(feature_name)

        # Sort uniques -------------------------------------------------------------------------------------
        if percent_null:
            try:
                uniques = tuple(np.append('nan', np.sort(uniques[uniques.notnull()])))
                unique_sorted = 'Yes'
            except TypeError:
                uniques = tuple(np.append('nan', np.sort(uniques[uniques.notnull()].astype(str))))
                unique_sorted = 'Partial'
        else:
            try:
                uniques = tuple(np.sort(uniques))
                unique_sorted = 'Yes'
            except TypeError:
                uniques = tuple(np.sort(uniques.astype(str)))
                unique_sorted = 'Partial'
        all_uniques_as_string = str(uniques)

        # Creating unique show -----------------------------------------------------------------------------
        if len(all_uniques_as_string) < unique_display_width:
            unique_show = f'{all_uniques_as_string[:unique_display_half]}' \
                          f'{all_uniques_as_string[unique_display_half:]}'
        else:
            unique_show = f'{all_uniques_as_string[:unique_display_half]} ... ' \
                          f'{all_uniques_as_string[-unique_display_half:]}'

        # Storing final report and dtype for the feature ---------------------------------------------------
        all_reports.append((problem, feature_name, percent_null, dtype, nuniques,
                            sum_value, mean_value, std_value, min_value, twenty_five_perc_value,
                            median_value, seventy_five_perc_value, max_value, unique_sorted,
                            ('%-' + str(unique_display_width-5) + 's') % unique_show))
        all_dtypes.append(dtype_suggest)
        print('|', end='')
        del current_feature

    print()
    final_report = pd.DataFrame(data=all_reports, columns=('!', 'name', '%null', 'dtype', 'nuniques', 'sum', 'mean',
                                                           'std', 'min', '0.25', 'median', '0.75', 'max', 'sorted?',
                                                           'uniques'))
    all_feature_types_suggested = dict(zip(all_features, all_dtypes))
    return final_report, all_feature_types_suggested, number_of_observations, numerical_features, \
        category_features, object_features, date_features


# IV. MODELING
def transform_main(x, list_of_drop, dict_of_poly, list_of_interactions,
                   all_interactions, all_poly_degree):

    error_check = False
    transformations_done = []

    if list_of_drop or dict_of_poly or list_of_interactions or all_interactions or \
            all_poly_degree > 1:

        poly_error = bool
        interaction_error = bool
        dropping_error = bool

        # Add polynomials
        if dict_of_poly:
            poly_error = transform_add_poly(x, dict_of_poly)
            transformations_done.append('add poly')

        # Add ALL possible polinomials (bringing the equation to quadratic form, only works if polies are not specified)
        if not poly_error and all_poly_degree > 1 and not dict_of_poly:
            transform_add_all_poly(x, all_poly_degree)
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
        if not interaction_error and not poly_error and list_of_drop:
            dropping_error = transform_drop_vars(x, list_of_drop)
            transformations_done.append('dropped custom variables')

        error_check = interaction_error and poly_error and dropping_error or False

    return error_check, transformations_done


def transform_add_interactions(x, list_of_interactions):
    for interaction in list_of_interactions:
        try:
            splitted_interact = interaction.split(':')
            x[interaction] = x[splitted_interact[0]] * x[splitted_interact[1]]
        except KeyError:
            print('One of the interactions is invalid. The custom model will be terminated.')
            return True
    return False


def transform_add_all_interactions(x):
    original_x_len = len(x.columns)
    for var in range(0, original_x_len):
        for next_var in range(var + 1, original_x_len):
            x[f'{x.columns[var]}:{x.columns[next_var]}'] = \
                x[x.columns[var]] * x[x.columns[next_var]]


def transform_add_poly(x, dict_of_polies):
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
    original_x_len = len(x.columns)
    for var_index in range(0, original_x_len):
        for degree_level in range(2, degree + 1):
            x[f"{x.columns[var_index]}^{degree_level}"] \
                = x[x.columns[var_index]] ** degree_level


def transform_drop_vars(x, list_of_vars_to_drop):
    if list_of_vars_to_drop:
        try:
            x.drop(list_of_vars_to_drop, axis=1, inplace=True)
        except KeyError:
            print('One of the vars to drop is invalid. The custom model will be terminated.')
            return True
    return False


def transform_get_dummies(df, categorical_features):
    df = df.categorize(categorical_features)
    dummy_encoder1 = DummyEncoder(columns=categorical_features, drop_first=False)
    dummy_encoder2 = DummyEncoder(columns=categorical_features, drop_first=True)
    dropped_features = set(dummy_encoder1.fit_transform(df).columns) ^ set(dummy_encoder2.fit_transform(df).columns)
    df = dummy_encoder2.fit_transform(df)

    return df, dropped_features


# REGRESSION
def regression():
    pass


# 2. ********************************************* EXECUTIONS ******************************************************* #
if __name__ == '__main__':
    main()
