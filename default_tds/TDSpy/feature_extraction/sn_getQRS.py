import numpy as np
import neurokit2 as nk

def sn_getQRS(signal : np.ndarray, sf = 200):
    """
    detects the R-Peaks in an ECG Signal \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param signal:  vector containing an ECG time series
    :param sf:      sampling frequency of time series
    :return:        list of sample indices of the R-Peaks
    """

    signals, info = nk.ecg_process(signal, sampling_rate=sf)
    rpeaks = info["ECG_R_Peaks"]

    return rpeaks