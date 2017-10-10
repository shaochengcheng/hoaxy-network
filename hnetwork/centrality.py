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


def k_core_evolution(fn, ofn=None, freq='D'):
    if ofn is None:
        ofn = 'k_core_evolution.csv'
    # load only necessary columns
    df = pd.read_csv(fn, parse_dates=['tweet_created_at'],
                     usecols=[2, 3, 4])
    df = df.set_index('tweet_created_at')
    # remove self-loop
    df = df.loc[df.from_raw_id != df.to_raw_id]
    df['row_id'] = np.arange(len(df))
    df['flag'] = False
    flag_rows = df.row_id.groupby(pd.Grouper(freq=freq)).last()
    flag_rows = flag_rows.loc[flag_rows.notnull()].astype('int')
    df.loc[df.row_id.isin(flag_rows.values), 'flag'] = True
    df = df.iloc[:100000]

    v_map = dict()
    e_set = set()
    v_counter = -1
    g = gt.Graph()
    mcore_k = []
    mcore_num = []
    ts = []
    for created_at, from_raw_id, to_raw_id, flag\
        in df[['from_raw_id', 'to_raw_id', 'flag']].itertuples():
        e = (from_raw_id, to_raw_id)
        if e not in e_set:
            if from_raw_id not in v_map:
                v_counter += 1
                v_map[from_raw_id] = v_counter
            if to_raw_id not in v_map:
                v_counter += 1
                v_map[to_raw_id] = v_counter
            source = v_map.get(from_raw_id)
            target = v_map.get(to_raw_id)
            g.add_edge(source, target, add_missing=True)
            e_set.add(e)
        if flag is True:
            ts.append(created_at)
            mcore = pd.Series(gt.kcore_decomposition(g).a.copy()).\
                value_counts().sort_index(ascending=False)
            mcore_k.append(mcore.index[0])
            mcore_num.append(mcore.iloc[0])
            logger.info('Main core at %s: k=%s, num=%s',
                        created_at, mcore.index[0], mcore.iloc[0])
    cdf = pd.DataFrame(dict(
        timeline=ts,
        mcore_k=mcore_k,
        mcore_num=mcore_num
    ))
    cdf.to_csv(ofn, index=False)



