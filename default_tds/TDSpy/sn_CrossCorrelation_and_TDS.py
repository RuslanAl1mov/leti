import itertools

import numpy as np

from TDSpy.sn_getCrossCorrelation import sn_getCrossCorrelation
from TDSpy.sn_getStability import sn_getStability


def sn_CrossCorrelation_and_TDS(res_dict, ws_sfe=1, wl_xcc=60, ws_xcc=30, wl_tds=5, ws_tds=1,
                                 mld_tds=2, mlf_tds=0.8):


    # check all possible combinations of signals in result dictionary
    combinations = list(itertools.combinations(res_dict.keys(), 2))

    first_value = True
    for comb in combinations:
        if first_value:
            xcc, xcl = sn_getCrossCorrelation(signal1=res_dict[comb[0]], signal2=res_dict[comb[1]], wl=wl_xcc,
                                              ws=ws_xcc, sf=1)
            stab = sn_getStability(xcl, wl=wl_tds, ws=ws_tds, mld=mld_tds, mlf=mlf_tds, sf=ws_sfe)
            tds = stab.reshape(1, -1)
            xcc = xcc.reshape(1, -1)
            xcl = xcl.reshape(1, -1)
            first_value = False
            # print('first:' , comb)
            # print(xcc.shape, xcl.shape)
        else:
            signal1 = res_dict[comb[0]]
            signal2 = res_dict[comb[1]]
            xcc_temp, xcl_temp = sn_getCrossCorrelation(signal1=res_dict[comb[0]], signal2=res_dict[comb[1]], wl=wl_xcc,
                                                        ws=ws_xcc, sf=ws_sfe)
            stab = sn_getStability(xcl_temp, wl=wl_tds, ws=ws_tds, mld=mld_tds, mlf=mlf_tds, sf=ws_sfe)
            tds = np.append(tds, stab.reshape(1, -1), axis=0)
            # print(comb)
            # print(xcc_temp.shape, xcl_temp.shape)
            xcc = np.append(xcc, xcc_temp.reshape(1, -1), axis=0)
            xcl = np.append(xcl, xcl_temp.reshape(1, -1), axis=0)

    return tds, combinations
