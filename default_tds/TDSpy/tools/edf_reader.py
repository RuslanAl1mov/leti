import numpy as np
from edfrd import read_header, read_data_records

# Available channels in SIESTA dataset:
# C3-M2, O1-M2, Fp2-M1, C4-M1, O2-M1, M2-M1, Pos8-M1, Pos18-M1,
# EMGmental, EMGleft, ECG, Airflow, Chest, Abdomen, SaO2

def read_single_EDF_channel(filepath : str, channel : str = "ECG", startrecord : int = 50, endrecord : int = 60):
    """
    reads a single chanel of an EDF file and outputs it in a vector\n
    Nico Spicher, 7.10.2022 \n
    Version 1.0

    :param filepath:    full path and filename of the EDF file
    :param channel:     chanel of EDF file to read
    :param startrecord: number of first record in the output vector set to 0 for the first record
    :param endrecord:   number of last record in the output vector. set to inf for the last record
    :return:            vector of timeseries of the channel
    """

    header = read_header(filepath, parse_date_time=True, calculate_number_of_data_records=True)
    wholeSignal = np.array([])

    if endrecord > header.number_of_data_records:
        endrecord = header.number_of_data_records

    # Iterate over all records and extract single from desired channel
    for data_record in read_data_records(filepath, header, start=startrecord, end = endrecord):
        for signal_header, signal in zip(header.signals, data_record):
            if signal_header.label == channel:
                wholeSignal = np.append(wholeSignal, signal, axis=0)
            else:
                continue

    return wholeSignal


def read_all_EDF_channels(filepath : str, startrecord : int = 50, endrecord : int = 60):
    """
    reads a full EDF file and outputs the timeseries in form of a dictionary and their sampling frequencies
    in a dictionary \n
    Nico Spicher, 7.10.2022 \n
    Version 1.0

    :param filepath:    full path and filename of the EDF file
    :param startrecord: number of first record in the output vector set to 0 for the first record
    :param endrecord:   number of last record in the output vector. set to inf for the last record
    :return:        data: dictionary containing the names of the channels as keys and their timeseries as values
                    data_sampling_rate: dictionary containing the names of the channels as keys and their timeseries
                    as values
    """

    header = read_header(filepath, parse_date_time=True, calculate_number_of_data_records=True)
    data = {}
    data_sampling_rate = {}
    first_run = True
    if endrecord > header.number_of_data_records:
        endrecord = header.number_of_data_records

    # Iterate over all records and extract single from desired channel
    for data_record in read_data_records(filepath, header, start=startrecord, end = endrecord):
        for signal_header, signal in zip(header.signals, data_record):
            if not first_run:
                data[signal_header.label] = np.concatenate((data[signal_header.label],signal), axis=0)
            else:
                data[signal_header.label] = signal
                data_sampling_rate[signal_header.label] = int(signal_header.nr_of_samples_in_each_data_record/header.duration_of_a_data_record)
        first_run = False

    return data, data_sampling_rate

