import numpy as np

def nld_movingMedian(data : np.ndarray, ns : int):
    """
    computes a median over given number of samples \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param data:    timeseries vector
    :param ns:      number of samples to calculate median from
    :return:        vector containing moving Median with same length as data
                    delay: number of samples, that are not averaged but constant
    """

    # calculate moving average

    # number of samples excluding actual sample
    ns = ns - 1

    # length of datarecord
    l = len(data)

    # if the data length is less than double the size of our window, we make the window half as long
    while l < 2 * ns:
        ns = int(ns / 2)

    # allocate buffer
    result = np.zeros(l - ns)

    for i in range(l-ns):
        result[i] = np.median(data[i:i+ns+1])

    # delay
    delay1 = np.floor(ns / 2).astype('int')
    delay2 = np.ceil(ns / 2).astype('int')
    # pad result with constant values (first and last averaged value)
    # todo if we take the same delay as in the matlab code we end up with a by one smaller vector
    result = np.concatenate((np.ones(delay1) * result[0], result, (np.ones(delay2) * result[-1])))

    return result, delay1
