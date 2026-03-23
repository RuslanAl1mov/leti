import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(matrix, thresh=7, labels=None, labels_color=None, save=None):
    G = nx.Graph()

    if labels is None:
        labels = ['HR', 'Resp', 'Chin', 'Leg', 'Eye', r'$\delta$', r'$\theta$', r'$\alpha$', r'$\sigma$', r'$\beta$']
    if labels_color is None:
        labels_color = ['tomato', 'dodgerblue', 'gold', 'silver', 'plum', 'limegreen', 'limegreen', 'limegreen',
                      'limegreen', 'limegreen']

    G.add_nodes_from(labels)
    pos = nx.circular_layout(G)

    for i in range(10):
        for j in range(i):
            if matrix[i, j] > thresh:
                G.add_edge(labels[i], labels[j])

    nx.draw(G, pos, with_labels=True, node_color=labels_color, node_size=800, font_size=10, font_color='k')
    if save is not None:
        plt.savefig(save)

    return
