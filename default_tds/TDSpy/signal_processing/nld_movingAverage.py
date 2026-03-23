import numpy as np

def nld_movingAverage(data : np.ndarray, ns):
    """
    computes an average over given number of samples \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0

    :param data: timeseries vector
    :param ns:   number of samples to be averaged
    :return:    vector containing moving average with same length as data
    """

    # calculate moving average

    # number of samples excluding actual sample
    ns = ns-1

    # length of datarecord
    l = len(data)

    # if the data length is less than double the size of our window, we make the window half as long    #todo
    while l < 2*ns:
        ns = int(ns/2)


    # allocate buffer
    result = np.zeros(l-ns)
    movStd = np.zeros(l-ns)

    # length of datarecord is supposed to be on dim 1
    for i in range(l-ns):
        #print('i = ', i, 'data = ', data[i:i+ns+1])
        result[i] = np.mean(data[i:i+ns+1])
        movStd[i] = np.std(data[i:i+ns+1])

    # delay
    delay1 = np.floor(ns/2).astype('int')
    delay2 = np.ceil(ns/2).astype('int')
    # pad result with constant values (first and last averaged value)
    # todo if we take the same delay as in the matlab code we end up with a by one smaller vector
    result = np.concatenate((np.ones(delay1)*result[0], result, (np.ones(delay2)*result[-1])))
    #print('result = ', result)
    return result, delay1, movStd
