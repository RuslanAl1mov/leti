import numpy as np

from TDSpy.nld_spectrogram import nld_spectrogram


def sn_getEEGBandPower(signal: np.ndarray, wl=2, ws=1, sf=200,
                       bandlimits=np.array([[0.5, 4, 8, 12, 16], [3.5, 7.5, 11.5, 15.5, 19.5]])):
    """
    calculates the overall power of EEG frequency bands \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param signal:      vector of EEG timeseries
    :param wl:          window length of fourier transform in seconds, default is 2 seconds
    :param ws:          window shift of fourier transform in seconds, default is 1 second
    :param sf:          sampling frequency of time series in Hz, default is 200 Hz
    :param bandlimits:  bandlimits matrix of shape (2, number_bands) where
                        bandlimits[0,n] = lower frequency limit of nth band
                        bandlimits[1,n] = upper frequency limit of nth band
    :return bp:         matrix containing the added power of frequency bands, with columns according to given bandlimits
    """

    # get spectrogram

    # samplingrate in seconds
    sr = 1/sf

    # windowlength in samples
    wls = wl*sf
    # windowshift in samples
    wss = ws*sf

    # get spectrogram
    # rows: frequency, columns: timeseries
    sbmat, f = nld_spectrogram(signal, samplingrate=sr, window_length=wls, window_shift=wss, dp=0)

    # get frequency resolution (difference between f[1]-f[0] = f[1]
    df = f[1]

    # get indices of band limits (+1 accounting for f[0] = 0)
    bandindices = (bandlimits/df)
    # for lower limit: ceil, for upper limit: floor
    bandindices[0, :] = np.ceil(bandindices[0, :])
    bandindices[1, :] = np.floor(bandindices[1, :])

    # Cumulative power of frequency bands

    # allocate buffer
    fpb = np.zeros((sbmat.shape[1], 5))

    # loop over bands
    for i in range(5):
        start = int(bandindices[0, i])
        end = int(bandindices[1, i])+1
        matrix_part = sbmat[start:end, :]
        fpb[:, i] = sum(matrix_part)

    return fpb, sbmat
