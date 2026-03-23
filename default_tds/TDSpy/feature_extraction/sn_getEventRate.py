import numpy as np
from scipy.interpolate import interp1d


def sn_getEventRate(events: np.ndarray, sf=1, out_len=15):
    """
    calculates the event rate from a vector conatining points of time of events \n
    Tabea Steinbrinker, 7.10.2022 \n
    Version 1.0


    :param events:  vector containing event times in sample units of the original signal
    :param sf:      sampling frequency of th original signal in Hz, default is 1 Hz
    :param out_len: desired length of output array
    :return:        eventrate vector conatining the event rate with desired length
    """

    # Get Event Rate

    # normalize sample points to a second
    events = events/sf
    # calculate event period (ep) in seconds: diff between event times
    ep = np.diff(events)
    # event rate is inverse values
    er = 1/ep
    # signal_length
    sl = max(events)

    epi = events[:-1] + ep / 2

    tv = np.linspace(0, sl, int(out_len) )

    inter = interp1d(epi, er, bounds_error=False, fill_value='extrapolate')(tv)

    return inter



    # # get low quality values
    # lqv = (qv <= lql)
    # # add preceding index with logical OR on shifted lqv
    # lqv = np.logical_or(lqv, np.concatenate((lqv[1:], np.array([False]))))
    # # delete event rate values derived from low quality values,
    # # also remove from events and ep, as they are further used to assign timepoints
    # er = np.delete(er, lqv[:-1])
    # events = np.delete(events, lqv)
    # ep = np.delete(ep, lqv[:-1])
    #
    # # generate vector with equidistant time points in seconds
    # tv = np.zeros((int(np.floor(sl*ersf/sf)), 2))
    # # points in time
    # timevector = np.linspace(0, len(tv)/ersf, len(tv)+1, endpoint=True)
    # tv[:, 0] = timevector[:len(tv)]
    #
    # if rsm == 'interval':
    #     #loop over all values
    #     for i in range(len(tv)):
    #         # check if timestamp is before first event
    #         if tv[i, 0] < events[0]:
    #             # assign first heart rate
    #             tv[i, 1] = er[0]
    #         else:
    #             time_diff = events - tv[i, 0]
    #             # number of negative values in time_diff
    #             events_index = sum(time_diff <= 0)
    #             tv[i, 1] = er[events_index]
    # elif rsm == 'interp':
    #     # time point index - time point to which the calculated period should be assigned:
    #     # middle point between the two time points the period is calculated from
    #     if tpa == 'interp':
    #         epi = events[:-1] + ep/2
    #     elif tpa == 'second':
    #         epi = events[1:]
    #     else:   #tpa == 'first'
    #         epi = events[:-1]
    #
    # # interpolate breathingperiods between first and last extrem value
    # tv[:, 1] = interp1d(epi, er, bounds_error=False, fill_value='extrapolate')(tv[:, 0])
    #
    # return tv[:,1]
