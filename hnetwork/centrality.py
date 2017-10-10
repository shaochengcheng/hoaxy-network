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


def centralities(g):
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    # degrees
    # in degree
    ki = g.degree_property_map('in')
    # out degree
    ko = g.degree_property_map('out')
    # weighted in degree
    si = g.degree_property_map('in', weight=g.ep['weight'])
    # weighted out degree
    so = g.degree_property_map('out', weight=g.ep['weight'])
    # pagerank
    pr = gt.pagerank(g)
    # betweetnness
    vb, eb = gt.betweenness(g)
    # eigenvector
    e, ev = gt.eigenvector(g)
    # screen_name
    sn = [g.vp['name'][v] for v in g.vertices()]
    df = pd.DataFrame(dict(
        screen_name=sn,
        in_degree=ki.a,
        out_degree=ko.a,
        weighted_in_degree=si.a,
        weighted_out_degree=so.a,
        page_rank=pr.a,
        betweenness=vb.a,
        eigenvector=ev.a
    ))
    df.to_csv('centralities.csv')


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


def k_core_evolution(fn, freq='D'):
    # load only necessary columns
    df = pd.read_csv(fn, parse_dates=['tweet_created_at'],
                     usecols=[2, 3, 4])
    df = df.set_index('tweet_created_at')
    # remove self-loop
    df = df.loc[df.from_raw_id != df.to_raw_id]
    ts = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    mcore_k = []
    mcore_num = []
    for d in ts[1:]:
        g = gt.Graph()
        ddf = df[:d].drop_duplicates()
        g.add_edge_list(ddf.values, hashed=True)
        gt.remove_parallel_edges(g)
        mcore = pd.Series(gt.kcore_decomposition(g).a.copy()).\
            value_counts().sort_index(ascending=False)
        mcore_k.append(mcore.index[0])
        mcore_num.append(mcore.iloc[0])



