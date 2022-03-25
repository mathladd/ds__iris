import pandas as pd
import glob

pd.set_option('display.expand_frame_repr', False)
FILE_LINK_LIST = glob.glob('~./*.csv')
LINK_TO_EXPORT_CSV = '~./'
SEP = ','


def check_data(data):
    """
    This is to filter out the data before passing to new csv file

    :param data: the data to be filtered out

    :return: Boolean True if conditions meet. The resulting data will be written into exporting file
    """

    # e.g.
    # if data[0] != 'USA' and data[3] != 0:
    #     return True

    if True or data[0] != 0:
        return True


def combine_with_headers(link_to_all_files, link_to_export, sep):
    """
    This is for combining multiple files that each all have headers

    :param link_to_all_files: link to all files to be combined
    :param link_to_export: like to the final export file
    :param sep: the separator of the data within each file (e.g. ',', ';', '\t')

    :return: None. The combined file can be found in specified folder
    """
    # If encoding eror, try encoding='Latin-1'
    count = 0
    with open(link_to_export, "a+", encoding="utf8") as targetfile:
        with open(link_to_all_files[0], "r", encoding="utf8") as f:
            for line in f:
                data = line.split(sep)
                if check_data(data):
                    targetfile.write(line)
                    count += 1
                    print(count)
        for file_link in link_to_all_files[1:]:
            with open(file_link, "r", encoding="utf8") as f:
                next(f)  # << only if the first line contains headers
                for line in f:
                    data = line.split(sep)
                    if check_data(data):
                        targetfile.write(line)
                        count += 1
                        print(count)


combine_with_headers(link_to_all_files=FILE_LINK_LIST, link_to_export=LINK_TO_EXPORT_CSV, sep=SEP)
