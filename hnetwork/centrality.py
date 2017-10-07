import logging

import graph_tool.all as gt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def load_graph(fn):
    return gt.load_graph_from_csv(fn,
                                  directed=True,
                                  eprop_types=('string', 'string', 'double'),
                                  eprop_names=('fromId', 'toId', 'weight'),
                                  string_vals=True,
                                  hashed=True,
                                  skip_first=True,
                                  ecols=(2,3),
                                  )

def v_betweenness(g):
    vb, eb = gt.betweenness(g)
    vdf = pd.DataFrame((g.vp['name'][i], vb[i])
                       for i in range(g.num_vertices()))
    vdf.index.name = 'v_idx'
    vdf.columns = ['screen_name', 'betweenness']
    edf = pd.DataFrame(dict(betweenness=eb.a))
    edf.index.name = 'e_idx'
    edf.columns = ['betweenness']
    vdf.to_csv('v_betweenness.csv')
    edf.to_csv('e_betweenness.csv')


def distance_histogram(g):
    import pdb;pdb.set_trace()
    counts, bins = gt.distance_histogram(g)
    df = pd.DataFrame(dict(counts=counts))
    df.to_csv('distance_histogram.csv', index=False)


def v_percolate(g, vertices, ofn):
    vertices = list(vertices)
    sizes, comp = gt.vertex_percolation(g, vertices)
    np.random.shuffle(vertices)
    sizes2, comp = gt.vertex_percolation(g, vertices)
    fig, ax = plt.subplots()
    ax.plot(sizes, label='Targeting')
    ax.plot(sizes2, label='Random')
    ax.set_xlabel("Vertices Remaining")
    ax.set_ylabel("Size of largest component")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ofn)

