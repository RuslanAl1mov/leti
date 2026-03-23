

import numpy as np
import matplotlib.pyplot as plt

def plot_TDS(tds, combinations, draw=True, show_numbers=False):
    length, width = tds.shape
    kopplung = np.sum(tds, axis=1)/width*100

    number_dict = {}
    i = 0
    for comb in combinations:
        for j in range(2):
            if comb[j] not in number_dict.keys():
                number_dict[comb[j]] = i
                i += 1

    to_plot = np.ones((i,i)) * 100
    for j, comb in enumerate(combinations):
        to_plot[number_dict[comb[0]], number_dict[comb[1]]] = kopplung[j]
        to_plot[number_dict[comb[1]], number_dict[comb[0]]] = kopplung[j]

    if draw:
        lbls = number_dict.keys()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(to_plot)
        im.set_clim(0, 100)
        fig.colorbar(im)

        ax.set_title('Kopplung der beiden Kanäle in Prozent')

        ax.set_xticks(np.arange(i), labels=lbls)
        ax.set_yticks(np.arange(i), labels=lbls)

        font = 5
        plt.setp(ax.get_yticklabels(), fontsize=font)
        plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right", fontsize=font)
        if show_numbers:
            for (j, i), label in np.ndenumerate(to_plot):
                ax.text(i, j, int(label), ha='center', va='center')


        plt.show()

    return to_plot



