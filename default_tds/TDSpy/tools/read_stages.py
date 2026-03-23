import numpy as np
import pandas as pd
import os

def read_stages(stages_file):

    df_sleepstages = pd.read_csv(stages_file, delim_whitespace=True, header=None, error_bad_lines=False)
    stages = df_sleepstages.to_numpy()[:, [0, 2]]

    stages_dict = {}
    beginning_time = stages[0,0]
    current_stage = stages[0,1]
    for counter in range(len(stages[:, 1])):
        if stages[counter, 1] == current_stage:
            current_time = stages[counter, 0]
        else:
            if current_stage not in stages_dict.keys():
                stages_dict[current_stage] = [(beginning_time, current_time)]
            else:
                stages_dict[current_stage].append((beginning_time, current_time))
            current_stage = stages[counter, 1]
            current_time = stages[counter, 0]
            beginning_time = stages[counter, 0]

    return stages_dict