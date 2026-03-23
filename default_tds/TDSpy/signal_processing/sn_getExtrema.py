import numpy as np


def sn_getExtrema(signal: np.ndarray, sf=4, mp=2):
    """
    gets all extreme values and then chooses the most prominent ones within a window of minimum period length \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param signal:  periodic time signal
    :param sf:      sampling frequency of time series in Hz, default is 4 Hz
    :param mp:      minimum period of signal in seconds, default is 2 sec
    :return:        extrema matrix of shape (n,4) with columns:
                    1: location of maxima
                    2: values maximum
                    3: location minimum
                    4: value minimum
    """

    # find maximum

    # windowlength in samples, where only one maximum should appear
    d = mp*sf

    # get size of dataset
    dim = len(signal)

    # get sign of gradients from time series
    signal_grad_sign = np.sign(np.diff(signal))

    # allocate buffer, based on minimum period
    nmax_extrema = np.ceil(len(signal)/d).astype('int')

    extrema_temp = np.zeros((nmax_extrema, 4))
    l = 0
    # loop over time
    for k in range(dim-2):
        # detect maxima
        # change from positive to zero or negative gradient
        if (signal_grad_sign[k] == 1) and (signal_grad_sign[k + 1] < 1):
            # the maximum sample is one ahead the gradient, therefore k+1
            extrema_temp[l, 0] = k+1
            extrema_temp[l, 1] = signal[k+1]
            l = l+1

        # detect minima
        # change from zero or negative gradient to positive gradient
        elif (signal_grad_sign[k] < 1) and (signal_grad_sign[k+1] == 1):
            if l < nmax_extrema:
                extrema_temp[l, 2] = k+1
                extrema_temp[l, 3] = signal[k+1]
            else:
                extrema_temp = np.concatenate((extrema_temp, np.array([[0, 0, k+1, signal[k+1]]])), axis=0 )

    # clip extrema_temp
    # get maximum index for maxima
    I = np.argmax(extrema_temp[:, 0])
    extrema_temp = extrema_temp[:I+1, :]

    # loop over all maxima
    m = 0
    # m must be smaller than the length of extrema
    while m < len(extrema_temp[:, 0])-1:        #added -1 here
        # look if distance is too short
        if extrema_temp[m+1, 0] - extrema_temp[m, 0] < d:
            # find lager maximum
            # case 1: fist maximum is larger
            if extrema_temp[m, 1] >= extrema_temp[m+1, 1]:
                # check for smaller minimum for following maximum
                if m+2 < len(extrema_temp[:, 0]):
                    if extrema_temp[m+1, 3] < extrema_temp[m+1, 2:3]:
                        # copy smaller minimum
                        extrema_temp[m+2, 2:3] = extrema_temp[m+1, 2:3]
                # delete smaller maximum value
                extrema_temp = np.delete(extrema_temp, m+1, axis=0)
            # case 2: second maximum is larger
            else:
                # both minima are possible, check for smaller minimum
                if extrema_temp[m, 3] < extrema_temp[m+1, 3]:
                    # copy smaller minimum
                    extrema_temp[m+1, 2:3] = extrema_temp[m, 2:3]
                extrema_temp = np.delete(extrema_temp, m, axis=0)
        else:
            m = m+1
    I = np.argmax(extrema_temp[:, 0])
    extrema = extrema_temp[:I+1, :]

    return extrema
