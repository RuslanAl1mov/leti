import numpy as np
import pandas as pd


def read_all_CSV_channels(filepath_data : str, filepath_keys : str):
    """
    reads two cvs files and outputs the timeseries in form of a dictionary and their sampling frequencies
    in a dictionary \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param filepath:    full path and filename of the CVS file

    :return:        data: dictionary containing the names of the channels as keys and their timeseries as values
                    data_sampling_rate: dictionary containing the names of the channels as keys and their timeseries
                    as values
    """

    df_keys = pd.read_csv(filepath_keys, sep=',', header=0)
    keys = df_keys.values[:, 1]
    list_keys = [key.strip() for key in keys]

    df_data = pd.read_csv(filepath_data, sep=',', header=0)
    array_data = df_data.to_numpy()

    data = {}
    data_sampling_rate = {}
    sampling_rate = int(1/array_data[1, 0])

    for i, key in enumerate(list_keys):
        data[key] = array_data[:, i+1]
        data_sampling_rate[key] = sampling_rate

    return data, data_sampling_rate
