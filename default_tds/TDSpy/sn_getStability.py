import numpy as np
import itertools
import math

def sn_getStability(signal : np.ndarray, wl=5, ws=1, sf=1, mld=2, mlf= 0.8):
    """
    calculates stable sequences in lag sequence \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param signal:  vector of timelags
    :param wl:      window length of sequence in seconds, default is 5 seconds
    :param ws:      window shift of sequence in seconds, default is 1 second
    :param sf:      sampling frequency of time series, default is 1
    :param mld:     maximum lag difference considered as stable, default is 2
    :param mlf:     minimum lag fraction in window that need to fulfill mld, default is 0.8
    :return:        binary vector, marking stable windows
    """

    # get window clips

    # windowlength in samples
    window_length = wl*sf

    signal_length = len(signal)
    if window_length > signal_length:
        window_length = signal_length

    # window_shift in samples
    window_shift = ws*sf
    # minimum lag number in samples
    minimum_lag_number = int(np.ceil(mlf*window_length))

    # get window number
    window_number = int(np.fix((signal_length-window_length)/window_shift))+1

    # allocate buffer for time-delay-stability array
    tds = np.zeros(window_number)

    # All combinations of minimum number of lags (mln) chosen from sequence (wl)
    window_indices = np.arange(window_length)
    # colum: indices, rows: combinations
    C = np.array(list(itertools.combinations(window_indices, minimum_lag_number)))
    # extend to meet all number of combinations
    n_comb = math.comb(window_length, minimum_lag_number)   # n choose k function
    # vectors for diffs
    diff_lags = np.zeros(n_comb)

    istart = -window_shift
    iend = 0
    # Window loop
    for iwin in range(window_number):
        istart = istart + window_shift
        iend = istart + window_length - 1
        # get signal window
        signal_clip = signal[istart:iend+1]
        for irow in range(n_comb):
            idx = C[irow, :]
            signal_clip_2 = signal_clip[idx]
            diff_lags[irow] = max(signal_clip_2) - min(signal_clip_2)
        if min(diff_lags) <= mld:
            tds[iwin] = 1

    # pad for symmetric windows
    # delay
    delay = int(window_length // 2)
    # remainder for padding at end
    srm = signal_length - iend + delay
    padend = max(0, int(srm // window_shift)-1 )

    tds = np.concatenate((np.ones(delay)*tds[0], tds, np.ones(padend)*tds[-1]))

    return tds
