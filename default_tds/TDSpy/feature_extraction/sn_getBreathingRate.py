import numpy as np
import neurokit2 as nk
from scipy import signal as sgn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from TDSpy.signal_processing.sn_getExtrema import sn_getExtrema
from TDSpy.signal_processing.nld_movingAverage import nld_movingAverage


def sn_getBreathingRate_2(signal: np.ndarray, sf=1, brsf=1):
    return 1

def sn_getBreathingRate(signal: np.ndarray, sf=1, rsf=4, mp=2, nsma=11, llma=0.5, brsf=1, out_len=15):

    """
    calculates the breathing rate \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param signal:  airflow signal
    :param sf:      sampling frequency of the singal in Hz, default is 1 Hz
    :param rsf:     resampling frequency, default is 2 Hz
    :param mp:      minimum period of signal, default is 2 secs
    :param nsma:    number of samples in moving average, default is 11 samples
    :param llma:    lower limit for accepted extreme values with respect to moving average
                    as fraction of the moving average, default is 0.5
    :param out_len: desired length of output array

    :return br:     breathingrate, resampled to fit an array of size out_len
            extrema: matrix containing the found extreme values
                    cols:
                    1: location maximum (in resampled signal)
                    2: values maximum
                    3: location minimum (in resampled signal)
                    4: value minimum
    """

    # Analyse signal

    # resample the signal to rsf
    signal_resampled = sgn.resample(signal, int(rsf/sf*len(signal)))

    # find extreme values with constrains on the minimum distance in tim between two maxima
    extrema = sn_getExtrema(signal_resampled, sf=rsf, mp=mp)

    #  extreme values with small amplitudes

    # get distance between maximum and minimum (amplitude)
    extrema_dist = extrema[:, 1] - extrema[:, 3]

    # get moving average of amplitudes
    extrema_dist_mavg, _, _ = nld_movingAverage(extrema_dist, nsma)

    # get extrema with distances smaller than lower limit
    extrema_dist_bin = (extrema_dist < llma * extrema_dist_mavg)

    # get extrema with larger maximum than preceding extremum
    extrema_max_ltp_bin = np.greater(extrema[:, 1], np.concatenate((np.array([0]), extrema[:-1, 1])))

    # get extrema with smaller minimum than following extremum
    extrema_min_ltf_bin = np.less(extrema[:, 3], np.concatenate((extrema[1:, 3], np.array([0]))))

    # before deleting false extrema, correct for larger maxima and smaller minima

    # false positives with smaller minima
    fpsm = np.logical_and(extrema_dist_bin, extrema_min_ltf_bin)

    # false positives with larger maxima
    fplm = np.logical_and(extrema_dist_bin, extrema_max_ltp_bin)

    # DEBUGGING
    # find those, where both conditions are given, they must be excluded from shifted, otherwise extrema order is not
    #   preserved
    fpsm_fplm = np.logical_and(fpsm, fplm)
    # invert to set incidents to false
    fpsm_fplm = ~fpsm_fplm
    # remove these from fpsm and fplm
    fpsm = np.logical_and(fpsm,fpsm_fplm)
    fplm = np.logical_and(fplm, fpsm_fplm)

    # store the minimal value to follower
    fpsm_temp_1 = np.concatenate((np.array([False]), fpsm[:-1]))
    fpsm_temp_2 = np.concatenate((fpsm[:-1], np.array([False])))
    extrema[fpsm_temp_1, 2:3] = extrema[fpsm_temp_2, 2:3]

    # store the maximal value to precessor
    fpsm_temp_1 = np.concatenate((np.array([False]), fpsm[1:]))
    fpsm_temp_2 = np.concatenate((fpsm[1:], np.array([False])))
    extrema[fpsm_temp_2, 0:1] = extrema[fpsm_temp_1, 0:1]

    # delete extreme values below fraction of moving average
    extrema = np.delete(extrema, extrema_dist_bin, axis=0)

    # delete also the moving average-values
    extrema_dist_mavg = np.delete(extrema_dist_mavg, extrema_dist_bin, axis=0)

    # and now the same story for the amplitudes between maxima and following minima

    # get distance between maximum and following minimum
    extrema_dist = extrema[:-1, 1]-extrema[1:, 3]

    # get extrema with distance smaller than lower limit
    edb = np.less(extrema_dist, llma*extrema_dist_mavg[:-1])

    # write minimum values from these false positives to follower
    edb_temp_1 = np.concatenate((np.array([False]), edb[:-1], np.array([False])))
    edb_temp_2 = np.concatenate((edb[:-1], np.array([False, False])))
    extrema[edb_temp_1, 2:3] = extrema[edb_temp_2, 2:3]

    # delete the false positives
    extrema = np.delete(extrema, np.concatenate((edb, np.array([False]))), axis=0)

    # Get breathing rate

    # calculate period: diff between minima and maxima
    bp = np.diff(extrema[:, [0, 2]], axis=0)
    bp = bp/rsf

    # moments to which the calculated period should be assigned:
    # middle point between the two time points the period is calculated from
    bpi = extrema[:-1, [0, 2]]+bp/2

    # flip cols for having the preceding minimum earlier than the following maximum
    bp = np.flip(bp, axis=1).reshape(len(bp)*2)
    bpi = np.flip(bpi, axis=1).reshape(len(bpi)*2)

    # kick out first and last value, as they correspond to length of first peak, not interpeak length
    bp = bp[1:-1]
    bpi = bpi[1:-1]

    tv = np.linspace(0, np.max(bpi) , int(out_len))

    # interpolate breathing periods between first and last extreme value
    # br = interp1d(bpi, )(np.linspace(int(np.ceil(bpi[0])), int(np.floor(bpi[-1]))))
    br = interp1d(bpi, (1./bp), bounds_error=False, fill_value='extrapolate')(tv)

    # # needs some correction
    # # pad at both ends to fit resampled data
    # padstart = int(np.ceil(bpi[0])-1)
    # padend = int(len(signal_resampled)-np.floor(bpi[-1]))+1
    # # if floor is 0, cut first point, not to get confused with indices and timepoints
    # if padstart == -1:
    #     breathingrate = np.concatenate((br[1:], np.ones(padend)*br[-1]))
    # else:
    #     breathingrate = np.concatenate((np.ones(padstart)*br[0], br, np.ones(padend)*br[-1]))
    #
    # breathingrate = sgn.resample(breathingrate, int(np.ceil(brsf/rsf*len(breathingrate))))

    return br, extrema, signal_resampled
