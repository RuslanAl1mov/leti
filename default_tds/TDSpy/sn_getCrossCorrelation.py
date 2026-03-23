import numpy as np
import scipy.signal as signal

def sn_getCrossCorrelation(signal1 : np.ndarray, signal2 : np.ndarray, wl : int = 30, ws : int = 30, sf : int = 1):
    """
    computes best cross correlation for signals, signal windows are normalized to zero mean and unit std \n
    Nicolai Spicher, 7.10.2022 \n
    Version 1.0

    :param signal1: first biosignal timeseries
    :param signal2: second biosignal timeseries
    :param wl:      window length of correlation in seconds, default is 60 seconds
    :param ws:      window shift of fourier transform in seconds, default is 30 seconds
    :param sf:      common sampling frequency of both signals
    :return:        xcc_signal: vector containing maximum correlation of each window
                    xcl_signal: corresponding lags of each window
    """

    # window length in samples
    window_length = wl * sf
    # window shift in samples
    window_shift = ws * sf

    try:
        np.shape(signal1) == np.shape(signal2)
    except:
        print("Signals are required to have the same number of samples N")




            # signal sizes
    signal_length = len(signal1)

    # calculate the number of windows that fit, consider signal-lengths, that are not multiples of window length,
    # and consider overlapping windows (window-shift is different from window length)
    window_number = np.fix((signal_length - window_length) / window_shift) + 1

    # allocate buffer
    xcc_signal, xcl_signal = np.zeros((2, int(window_number)))

    istart = 1 - window_shift - 1
    for iwin in range(int(window_number)):

        istart = istart + window_shift
        iend = istart + window_length
        # get signal window
        signal_clip1 = signal1[istart:iend].copy()
        signal_clip2 = signal2[istart:iend].copy()
        # subtract offset
        signal_clip1 -= np.mean(signal_clip1)
        signal_clip2 -= np.mean(signal_clip2)
        # normalize to unit standard-deviation
        if np.std(signal_clip1) != 0:
            signal_clip1 /= np.std(signal_clip1, ddof=1)
        if np.std(signal_clip2) != 0:
            signal_clip2 /= np.std(signal_clip2, ddof=1)

        # correlation
        ccx_clip = signal.correlate(signal_clip1, signal_clip2, mode="full")
        lags_clip = signal.correlation_lags(len(signal_clip1), len(signal_clip2), mode="full")

        if np.isnan(ccx_clip).all():
            xcc_signal[iwin] = np.nan
            xcl_signal[iwin] = np.nan
            continue

        # store
        xcc_signal[iwin] = np.max(np.abs(ccx_clip))
        # take all the indices, where the maximum is attained and choose the one, where the abs value of lags_clip is
        # minimal
        max_ccx_clip_idx = np.argwhere(np.abs(ccx_clip) == np.amax(np.abs(ccx_clip))).flatten()
        rel_idx = max_ccx_clip_idx[np.argmin(np.abs(lags_clip[max_ccx_clip_idx]))]
        xcl_signal[iwin] = lags_clip[rel_idx]
        # xcl_signal[iwin] = lags_clip[np.argmax(np.abs(ccx_clip))]

        # sometimes, cross-correlation does detect the wrong time-shift due to random fluctuations looking similar
        # use: iwin>0 and np.max(xcl_signal[0:iwin+1]) != np.min(xcl_signal[0:iwin+1]) find these instances

    # pad crosscorrelation matrices in case of larger window_length than window_shift
    # get signalremainder not being windowed
    srm = signal_length - iend
    # get number of windowshifts that would fit into the remainder
    padnumber = int(np.floor(srm / window_shift))

    if padnumber > 0:
        # pad last value
        xcc_signal = np.concatenate((xcc_signal, np.ones(padnumber)*xcc_signal[-1]) )
        xcl_signal = np.concatenate((xcl_signal, np.ones(padnumber)*xcl_signal[-1]) )

    return xcc_signal, xcl_signal