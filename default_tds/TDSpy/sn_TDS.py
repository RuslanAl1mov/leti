import itertools

import numpy as np
import pandas as pd
from scipy.signal import resample

from TDSpy.feature_extraction.sn_getBreathingRate import sn_getBreathingRate
from TDSpy.feature_extraction.sn_getEEGBandPower import sn_getEEGBandPower
from TDSpy.feature_extraction.sn_getEventRate import sn_getEventRate
from TDSpy.feature_extraction.sn_getQRS import sn_getQRS
from TDSpy.feature_extraction.sn_getVariance import sn_getVariance
from TDSpy.signal_processing.nld_movingMedian import nld_movingMedian
from TDSpy.sn_getCrossCorrelation import sn_getCrossCorrelation
from TDSpy.sn_getStability import sn_getStability
from TDSpy.tools.edf_reader import read_all_EDF_channels


def sn_TDS(file=None, montage=None, data_dict=None, data_dict_sampling_rate=None, wl_sfe=2, ws_sfe=1,
           wl_xcc=60, ws_xcc=30, wl_tds=5, ws_tds=1, mld_tds=2, mlf_tds=0.8, startrecord=50, endrecord=60 ):
    """
    reads a file in EDF format or takes a data dict and performs TDS method \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0


    :param file:        full filename of an EDF file, including path
    :param montage:     dictionary containing the names of the channels as keys and their type as value as strings.
                        Possibilities are EEG, EOG, EMG, ECG, and Resp
    :param data_dict:   dictionary of the data, where the keys are the names of the channels as strings and the values
                        are the timeseries,
                        only needed if file=None
    :param data_dict_sampling_rate: dictionary of the sampling rates of the data, where the keys are the names of the
                        channels as strings and the values are the timeseries,
                        only needed if file=None
    :param wl_sfe:      windowlength of signal feature extraction in seconds, default is 2 seconds
    :param ws_sfe:      windowshift of signal feature extraction in seconds, default is 1 second
    :param wl_xcc:      windowlength of crosscorrelation in seconds, default is 60 seconds
    :param ws_xcc:      windowshift of crosscorrelation in seconds, default is 30 seconds
    :param wl_tds:      windowlength of stability analysis in seconds, default is 5 seconds
    :param ws_tds:      windowshift of stability analysis in seconds, default is 1 second
    :param mld_tds:     maximum lag difference in window to be accounted as stable sequence, default is 2
    :param mlf_tds:     minimum lag fraction in window that need to fulfill mld_tds, default is 0.8
    :param startrecord: point to start reading of the EDF file, for beginning take 0
    :param endrecord:   point to end reading of the EDF file, for end take inf
    :return:            matrix containing stability matrix of size(timespan, nsignals^2)
                        the time resolution is determined by ws_xcc. The order of the order of the signal2signal is
                        first all combinations with first signal, all combinations with second signal, ..., all signals
                        with n-th signal. eg for three signals: 12 - 13 - 23
    """

    if montage is None:
        montage = {'Fp1-M2': 'EEG', 'C3-M2': 'EEG', 'O1-M2': 'EEG', 'Fp2-M1': 'EEG', 'C4-M1': 'EEG', 'O2-M1': 'EEG', 'M2-M1': 'EEG',
                   'Pos8-M1': 'EEG', 'Pos18-M1': 'EEG',
                   'EMGmental': 'EMG', 'EMGleft': 'EMG', 'ECG': 'ECG', 'Airflow': 'Resp', 'Chest': 'Resp', 'Abdomen': 'Resp'}
    #,
    if file != None:
        # read edf
        data_dict, data_dict_sampling_rate = read_all_EDF_channels(file, startrecord=startrecord, endrecord=endrecord)

    # define result dictionary
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
            rrdata = sn_getQRS(data_dict[key], sf=data_dict_sampling_rate[key])
            sl_hr = len(data_dict[key])
            sf_hr = data_dict_sampling_rate[key]
            sample_number = int((np.floor((sl_hr - 1) / (ws_sfe * sf_hr))) + 1)
            # get heart rate
            heartrate = sn_getEventRate(rrdata, sf=data_dict_sampling_rate[key], out_len=sample_number)
            heartrate = nld_movingMedian(heartrate, 5)[0]
            res_dict[key] = heartrate

        elif montage[key] == 'Resp':
            sl_hr = len(data_dict[key])
            sf_hr = data_dict_sampling_rate[key]
            sample_number = int((np.floor((sl_hr - 1) / (ws_sfe * sf_hr))) + 1)
            breathingrate = sn_getBreathingRate(data_dict[key], sf=data_dict_sampling_rate[key], brsf=ws_sfe, out_len=sample_number)[0]
            res_dict[key] = breathingrate

        elif montage[key] == 'Raw':
            sl_hr = len(data_dict[key])
            sf_hr = data_dict_sampling_rate[key]
            sample_number = int((np.floor((sl_hr - 1) / (ws_sfe * sf_hr))) + 1)
            signal = data_dict[key]
            res_dict[key] = resample(signal, sample_number)
            # res_dict[key] = signal

    # check all possible combinations of signals in result dictionary
    combinations = list(itertools.combinations(res_dict.keys(), 2))


    first_value = True
    for comb in combinations:
        if first_value:
            xcc, xcl = sn_getCrossCorrelation(signal1=res_dict[comb[0]], signal2=res_dict[comb[1]], wl=wl_xcc, ws=ws_xcc, sf=ws_sfe)
            stab = sn_getStability(xcl, wl=wl_tds, ws=ws_tds, mld=mld_tds, mlf=mlf_tds, sf=ws_sfe)
            tds = stab.reshape(1, -1)
            xcc = xcc.reshape(1, -1)
            xcl = xcl.reshape(1, -1)
            first_value = False
            #print('first:' , comb)
            #print(xcc.shape, xcl.shape)
        else:

            signal1 =res_dict[comb[0]]
            signal2 = res_dict[comb[1]]
            xcc_temp, xcl_temp = sn_getCrossCorrelation(signal1=res_dict[comb[0]], signal2=res_dict[comb[1]], wl=wl_xcc, ws=ws_xcc, sf=ws_sfe)
            stab = sn_getStability(xcl_temp, wl=wl_tds, ws=ws_tds, mld=mld_tds, mlf=mlf_tds, sf=ws_sfe)
            tds = np.append(tds, stab.reshape(1,-1), axis=0)
            #print(comb)
            #print(xcc_temp.shape, xcl_temp.shape)
            xcc = np.append(xcc, xcc_temp.reshape(1, -1), axis=0)
            xcl = np.append(xcl, xcl_temp.reshape(1, -1), axis=0)

    return tds, combinations


def sn_TDS_no_feature_extraction(data_dict=None, ws_sfe=1, wl_xcc=60, ws_xcc=30, wl_tds=5, ws_tds=1,
                                 mld_tds=2, mlf_tds=0.8, stages={}):
    """
    reads a file in EDF format or takes a data dict and performs TDS method \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0


    :param file:        full filename of an EDF file, including path
    :param montage:     dictionary containing the names of the channels as keys and their type as value as strings.
                        Possibilities are EEG, EOG, EMG, ECG, and Resp
    :param data_dict:   dictionary of the data, where the keys are the names of the channels as strings and the values
                        are the timeseries,
                        only needed if file=None
    :param data_dict_sampling_rate: dictionary of the sampling rates of the data, where the keys are the names of the
                        channels as strings and the values are the timeseries,
                        only needed if file=None
    :param wl_sfe:      windowlength of signal feature extraction in seconds, default is 2 seconds
    :param ws_sfe:      windowshift of signal feature extraction in seconds, default is 1 second
    :param wl_xcc:      windowlength of crosscorrelation in seconds, default is 60 seconds
    :param ws_xcc:      windowshift of crosscorrelation in seconds, default is 30 seconds
    :param wl_tds:      windowlength of stability analysis in seconds, default is 5 seconds
    :param ws_tds:      windowshift of stability analysis in seconds, default is 1 second
    :param mld_tds:     maximum lag difference in window to be accounted as stable sequence, default is 2
    :param mlf_tds:     minimum lag fraction in window that need to fulfill mld_tds, default is 0.8
    :param startrecord: point to start reading of the EDF file, for beginning take 0
    :param endrecord:   point to end reading of the EDF file, for end take inf
    :return:            matrix containing stability matrix of size(timespan, nsignals^2)
                        the time resolution is determined by ws_xcc. The order of the order of the signal2signal is
                        first all combinations with first signal, all combinations with second signal, ..., all signals
                        with n-th signal. eg for three signals: 12 - 13 - 23
    """




    # define result dictionary
    res_dict = data_dict

    # check all possible combinations of signals in result dictionary
    combinations = list(itertools.combinations(res_dict.keys(), 2))


    first_value = True
    for comb in combinations:
        if first_value:
            xcc, xcl = sn_getCrossCorrelation(signal1=res_dict[comb[0]], signal2=res_dict[comb[1]], wl=wl_xcc, ws=ws_xcc, sf=1)
            stab = sn_getStability(xcl, wl=wl_tds, ws=ws_tds, mld=mld_tds, mlf=mlf_tds, sf=ws_sfe)
            tds = stab.reshape(1, -1)
            xcc = xcc.reshape(1, -1)
            xcl = xcl.reshape(1, -1)
            first_value = False
            #print('first:' , comb)
            #print(xcc.shape, xcl.shape)
        else:

            signal1 =res_dict[comb[0]]
            signal2 = res_dict[comb[1]]
            xcc_temp, xcl_temp = sn_getCrossCorrelation(signal1=res_dict[comb[0]], signal2=res_dict[comb[1]], wl=wl_xcc, ws=ws_xcc, sf=ws_sfe)
            stab = sn_getStability(xcl_temp, wl=wl_tds, ws=ws_tds, mld=mld_tds, mlf=mlf_tds, sf=ws_sfe)
            tds = np.append(tds, stab.reshape(1,-1), axis=0)
            #print(comb)
            #print(xcc_temp.shape, xcl_temp.shape)
            xcc = np.append(xcc, xcc_temp.reshape(1, -1), axis=0)
            xcl = np.append(xcl, xcl_temp.reshape(1, -1), axis=0)

    if bool(stages):
    
        stages_tds = np.empty(len(tds[0,:]), dtype=object)
        time_begin = -ws_xcc
        counter_table = 0
        
        for counter in range(len(tds[0,:])):
                
            time_begin = time_begin + ws_xcc
            time_end = time_begin + (wl_tds-1)*ws_xcc + wl_xcc

            stages_tds[counter] = np.nan
            
            for key in stages.keys():
                if any(lower <= time_begin <= upper for (lower, upper) in stages[key]):
                    stages_tds[counter_table] = key
                    counter_table = counter_table + 1                 
                    break

    else:
        stages_tds = None


    return tds, combinations, stages_tds
