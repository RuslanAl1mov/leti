import itertools

import numpy as np
import pandas as pd
from scipy.signal import resample
import neurokit2 as nk

from TDSpy.feature_extraction.sn_getBreathingRate import sn_getBreathingRate
from TDSpy.feature_extraction.sn_getEEGBandPower import sn_getEEGBandPower
from TDSpy.feature_extraction.sn_getEventRate import sn_getEventRate
from TDSpy.feature_extraction.sn_getQRS import sn_getQRS
from TDSpy.feature_extraction.sn_getVariance import sn_getVariance
from TDSpy.signal_processing.nld_movingMedian import nld_movingMedian
from TDSpy.sn_getCrossCorrelation import sn_getCrossCorrelation
from TDSpy.sn_getStability import sn_getStability
from tools.edf_reader import read_all_EDF_channels

def feature_extraction(data_dict, montage, data_dict_sampling_rate,  wl_sfe=2, ws_sfe=1):

    #Process all channels, by their type (ECG, EEG, etc.) given in montage
    res_dict = {}

    for key in montage:
        if montage[key] == 'EEG':
            fpb, sbmat = sn_getEEGBandPower(data_dict[key], wl=wl_sfe, ws=ws_sfe, sf=data_dict_sampling_rate[key])
            for i, band in enumerate(['_delta', '_theta', '_alpha', '_sigma', '_beta']):
                res_dict[key + band] = np.sqrt(fpb[:,i])

        elif montage[key] == 'EOG' or montage[key] == 'EMG':
            var = sn_getVariance(data_dict[key], wl=wl_sfe, ws=ws_sfe, sf=data_dict_sampling_rate[key])
            res_dict[key] = var

        elif montage[key] == 'ECG':
            #process ECG with help of neurokit
            ecg_signal = nk.ecg_process(data_dict[key], sampling_rate=data_dict_sampling_rate[key],
                                    method='neurokit')
            hr_signal_resampled = nk.signal_resample(ecg_signal[0]['ECG_Rate'], sampling_rate=data_dict_sampling_rate[key],
                                    desired_sampling_rate=1, method="interpolation")
            res_dict[key] = hr_signal_resampled

        elif montage[key] == 'Resp':
            #process Respiration signal with helf of neurokit
            rsp_rate = nk.rsp_rate(data_dict[key], sampling_rate=8, method="trough")
            rsp_signal_resampled = nk.signal_resample(rsp_rate, sampling_rate=data_dict_sampling_rate[key],
                                    desired_sampling_rate=1, method="interpolation")
            res_dict[key] = rsp_signal_resampled

        elif montage[key] == 'Raw':
            res_dict[key] = nk.signal_resample(data_dict[key], sampling_rate=data_dict_sampling_rate[key],
                                    desired_sampling_rate=1, method="interpolation")


    return res_dict