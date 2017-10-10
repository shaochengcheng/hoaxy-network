import logging

import graph_tool.all as gt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def weight_edge_list(fn, ofn):
    df = pd.read_csv(fn)
    df = df.loc[df.from_raw_id != df.to_raw_id]
    weight = df.groupby(['from_raw_id', 'to_raw_id']).size().rename('weight')
    weight.to_csv(ofn, index=True)


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
    df['gpf'] = False
    gpf_rows = df.row_id.groupby(pd.Grouper(freq=freq)).last()
    gpf_rows = gpf_rows.loc[gpf_rows.notnull()].astype('int')
    df.loc[df.row_id.isin(gpf_rows.values), 'gpf'] = True

    v_map = dict()
    e_set = set()
    v_counter = -1
    g = gt.Graph()
    mcore_k = []
    mcore_num = []
    mcore_idx = []
    ts = []
    for created_at, from_raw_id, to_raw_id, gpf\
        in df[['from_raw_id', 'to_raw_id', 'gpf']].itertuples():
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
        if gpf:
            ts.append(created_at)
            kcore = pd.Series(gt.kcore_decomposition(g).a.copy())
            mcore = kcore.value_counts().sort_index(ascending=False)
            mk = mcore.index[0]
            mn = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_num.append(mn)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            logger.info(g)
            logger.info('Main core at %s: k=%s, num=%s',
                        created_at, mk, mn)
    cdf = pd.DataFrame(dict(
        timeline=ts,
        mcore_k=mcore_k,
        mcore_num=mcore_num,
        mcore_idx=mcore_idx,
    ))
    cdf.to_csv(ofn, index=False)
    v_series = pd.Series(v_map)
    v_series.to_csv('vertex_map.csv')


def plot_kcore_timeline(fn='k_core_evolution.csv'):
    df = pd.read_csv(fn, parse_dates=['timeline'])
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.plot(df.timeline.values, df.mcore_k, label='K of Main Core', color='b')
    ax.plot(df.timeline.values, df.mcore_num, label='Number of Main core', color='r')
    plt.tight_layout()
    plt.savefig('k-core.pdf')


def changes_of_cores(fn1='k_core_evolution.csv', fn2='vertex_map.csv'):
    df1 = pd.read_csv(fn1, parse_dates=['timeline'])
    df2 = pd.read_csv(fn2, header=None)
    df2.columns = ['raw_id', 'v_idx']
    df2 = df2.set_index('v_idx')
    df1['mcore_idx'] = df1.mcore_idx.apply(eval).apply(set)
    df1 = df1.set_index('timeline')
    s1 = df1.loc['2016-11-07', 'mcore_idx'].iloc[0]
    s2 = set(s1)
    for ts, s0 in df1.mcore_idx.loc['2016-11-08':].iteritems():
        s1 &= s0
        s2 |= s0
    unchanged = df2.loc[list(s1)]
    unions = df2.loc[list(s2)]
    unchanged.to_csv('unchanged.csv')
    unions.to_csv('unions.csv')
    logger.info('Number of unchanged is %s', len(unchanged))
    logger.info('Number of union is %s', len(unions))


def main_core(df):
    w = df.groupby(['from_raw_id', 'to_raw_id']).size()
    g = gt.Graph()
    v_raw_ids = g.add_edge_list(w.index.values, hashed=True).a.copy()
    kcores = gt.kcore_decomposition(g).a.copy()
    s = pd.Series(kcores)
    return v_raw_ids[s.loc[s==s.max()].index]


def rolling_k_core(fn='retweet.201710.claim.raw.csv'):
    df = pd.read_csv(fn, parse_dates=['tweet_created_at'],
                     usecols=[2, 3, 4])
    df = df.set_index('tweet_created_at')
    # remove self-loop
    df = df.loc[df.from_raw_id != df.to_raw_id]
    df1 = df.loc[:'2016-11-07']
    df2 = df.loc['2016-11-08':'2017-04-07']
    df2 = df.loc['2017-04-08':]
    logger.info('Dataset 1: rows %s, days %s',
                len(df1), df1.index.max()-df1.index.min())
    logger.info('Dataset 2: rows %s, days %s',
                len(df2), df2.index.max()-df2.index.min())
    logger.info('Dataset 3: rows %s, days %s',
                len(df3), df3.index.max()-df3.index.min())
    w1 = df1.groupby(['from_raw_id', 'to_raw_id']).size()
    w2 = df2.groupby(['from_raw_id', 'to_raw_id']).size()
    w3 = df3.groupby(['from_raw_id', 'to_raw_id']).size()
    g1 = gt.Graph()
    g2 = gt.Graph()
    g3 = gt.Graph()
    g1.add_edge_list(w1.index.values, hashed=True)
    g2.add_edge_list(w2.index.values, hashed=True)
    g3.add_edge_list(w3.index.values, hashed=True)
    s1 = pd.Series(gt.kcore_decomposition(g1).a.copy())
    s2 = pd.Series(gt.kcore_decomposition(g2).a.copy())
    s2 = pd.Series(gt.kcore_decomposition(g3).a.copy())





