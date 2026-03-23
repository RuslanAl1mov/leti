import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt


def nld_spectrogram(signal: np.ndarray, samplingrate=1, window_length: int = 512, window_shift=100, mainfrequency=0,
                    cb=0, ha=1, sc='ln', ft='Spectogram of data', dp=True ):
    """
    produces shortterm Fourier Transforms of timeseries  \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param signal:          vector containing time series of data
    :param samplingrate:    sampling rate in seconds
    :param window_length:   window length in samples
    :param window_shift:    window shift in samples
    :param mainfrequency:   main frequency in Hz
    :param cb:              colorbar option, 1:'on', 0:'off'
    :param ha:              windowing: flattop='ft', hann='ha', default=1: no window
    :param sc:              scaling: linear='ln', default = log10
    :param ft:              figure title, default:'Spectrogram of title
    :param dp:              display, off=0, on=1

    :return sbmat:          matrix containing the absolut values of shortterm ffts
    :return f:              vector containing the frequency-values
    """

    if ha == 'ha':
        hw = np.hanning(window_length)
    elif ha == 'ft':
        hw = sgn.windows.flattop(window_length)
    else:
        hw = 1

    # spectrogram produces shortterm ffts of timeseries
    signal_length = len(signal)

    # subtraction of dc
    signal = signal - np.mean(signal)

    # cut redundant spectrum part
    spectrum_length = int(window_length//2)

    # calculate number of shortterm fft
    window_number = int((signal_length-window_length)//window_shift +1)

    # get sample number (number of samples with windowshift und unity wl
    sample_number = int((signal_length - 1) // window_shift + 1)

    # allocate matrix for result
    sbmat = np.zeros((spectrum_length, window_number))

    istart = - window_shift

    for iwin in range(window_number):
        istart = istart + window_shift
        iend = istart + window_length
        # multiply with window
        signal_clip = signal[istart:iend]
        signal_window = signal_clip*hw
        # fourier transform
        ft_t = np.fft.fft(signal_window)
        # power spectrum
        pspec = ft_t * np.conj(ft_t)
        # set zero of lowest value
        pspec[0] = min(pspec[1:spectrum_length])
        # cut redundant part
        pspec = pspec[: spectrum_length]

        # result
        sbmat[:, iwin] = pspec

    # delay
    delay = window_length // (2*window_shift)
    remains = sample_number - delay - window_number

    sbmat = np.concatenate((np.ones((spectrum_length, delay)) * sbmat[:, 0].reshape(spectrum_length, 1), sbmat,
                            np.ones((1, remains))*sbmat[:, -1].reshape(spectrum_length, 1)), axis=1)

    f = np.arange(0, spectrum_length)/(samplingrate *2* spectrum_length)   #*2

    if dp:
        freqmax = (spectrum_length - 1) / (2 * spectrum_length) / samplingrate
        x = np.array([0, window_number * window_shift * samplingrate])
        y = np.array([0, freqmax])

        fig = plt.figure()
        if sc == 'ln':
            plt.show(sbmat)
        else:
            plt.semilogy(sbmat)

        plt.ylabel('f[Hz]')
        plt.xlabel('t[x]')

        if cb:
            fig.colorbar()

        plt.show()

    return sbmat, f
