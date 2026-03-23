import numpy as np

def sn_getVariance(signal : np.ndarray, sf=200, wl=2, ws=1):
    """
    computes the variance of the signal in moving windows \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param signal:  time series vector
    :param sf:      sampling frequency of the time series
    :param wl:      window length to compute variance from in seconds
    :param ws:      window shift
    :return:        vector containing variance for the chosen windows
    """

    # get the variance
    # sampling rate in seconds
    sr = 1/sf

    # window length and window shift in samples
    window_length = wl*sf
    window_shift = ws*sf

    signal_length = len(signal)

    # get window number
    window_number = int(np.floor((signal_length-window_length)/window_shift) + 1)
    # get sampling number
    sample_number = int(np.floor((signal_length - 1)/window_shift) + 1)

    result = np.array([np.var(signal[iwin*window_shift : iwin*window_shift+window_length]) for iwin in range(window_number)])

    # correct array length for delay
    delay = int(np.floor(window_length/(2*window_shift)))

    # samples to be added at the end
    remains = int(sample_number - delay - window_number)

    # pad the result with constant values (first and last average value)
    result = np.concatenate((np.ones(delay) * result[0], result, (np.ones(remains) * result[-1])))

    return result