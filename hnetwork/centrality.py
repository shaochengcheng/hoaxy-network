import logging

import graph_tool.all as gt
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
    return gt.load_graph_from_csv(
        fn,
        directed=True,
        eprop_types=('string', 'string', 'double'),
        eprop_names=('fromId', 'toId', 'weight'),
        string_vals=True,
        hashed=True,
        skip_first=True,
        ecols=(2, 3),)


def prepare_network_from_raw(fn):
    df = pd.read_csv(fn, usecols=[3, 4])
    df = df.loc[df.from_raw_id != df.to_raw_id]
    w = df.groupby(['from_raw_id',
                    'to_raw_id']).size().rename('weight').reset_index(
                        drop=False)
    g = gt.Graph()
    v_raw_ids = g.add_edge_list(
        w[['from_raw_id', 'to_raw_id']].values, hashed=True)
    g.vp['raw_id'] = v_raw_ids
    e_weight = g.new_edge_property("double")
    e_weight.a = w['weight'].values
    g.ep['weight'] = e_weight
    return g


def centralities(g, user_map):
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
    screen_name = user_map.loc[g.vp['raw_id'].a.copy()].values
    df = pd.DataFrame(
        dict(
            screen_name=screen_name,
            in_degree=ki.a,
            out_degree=ko.a,
            weighted_in_degree=si.a,
            weighted_out_degree=so.a,
            page_rank=pr.a,
            betweenness=vb.a,
            eigenvector=ev.a))
    df.to_csv('centralities.raw.csv')
    df = df.sort_values('in_degree', ascending=False)
    ranked_ki = df.screen_name.values.copy()
    df = df.sort_values('out_degree', ascending=False)
    ranked_ko = df.screen_name.values.copy()
    df = df.sort_values('weighted_in_degree', ascending=False)
    ranked_si = df.screen_name.values.copy()
    df = df.sort_values('weighted_out_degree', ascending=False)
    ranked_so = df.screen_name.values.copy()
    df = df.sort_values('page_rank', ascending=False)
    ranked_pr = df.screen_name.values.copy()
    df = df.sort_values('betweenness', ascending=False)
    ranked_bt = df.screen_name.values.copy()
    df = df.sort_values('eigenvector', ascending=False)
    ranked_ev = df.screen_name.values.copy()
    ranked_df = pd.DataFrame(
        dict(
            in_degree=ranked_ki,
            out_degree=ranked_ko,
            weighted_in_degree=ranked_si,
            weighted_out_degree=ranked_so,
            page_rank=ranked_pr,
            betweenness=ranked_bt,
            eigenvector=ranked_ev))
    ranked_df.to_csv('centralities.ranked.csv', index=False)


def rank_centralities(fn1='centralities.raw.csv',
                      fn2='retweet.1108.claim.vmap.csv'):
    df = pd.read_csv(fn1)
    vmap = pd.read_csv(fn2)
    vmap = vmap.raw_id
    df = df.sort_values('in_degree', ascending=False)
    ranked_ki = df.screen_name.values.copy()
    ranked_kii = vmap.loc[df.index.tolist()].values.copy()
    ranked_kiv = df.in_degree.values.copy()
    df = df.sort_values('out_degree', ascending=False)
    ranked_ko = df.screen_name.values.copy()
    ranked_koi = vmap.loc[df.index.tolist()].values.copy()
    ranked_kov = df.out_degree.values.copy()
    df = df.sort_values('weighted_in_degree', ascending=False)
    ranked_si = df.screen_name.values.copy()
    ranked_sii = vmap.loc[df.index.tolist()].values.copy()
    ranked_siv = df.weighted_in_degree.values.copy()
    df = df.sort_values('weighted_out_degree', ascending=False)
    ranked_so = df.screen_name.values.copy()
    ranked_soi = vmap.loc[df.index.tolist()].values.copy()
    ranked_sov = df.weighted_out_degree.values.copy()
    df = df.sort_values('page_rank', ascending=False)
    ranked_pr = df.screen_name.values.copy()
    ranked_pri = vmap.loc[df.index.tolist()].values.copy()
    ranked_prv = df.page_rank.values.copy()
    df = df.sort_values('betweenness', ascending=False)
    ranked_bt = df.screen_name.values.copy()
    ranked_bti = vmap.loc[df.index.tolist()].values.copy()
    ranked_btv = df.betweenness.values.copy()
    df = df.sort_values('eigenvector', ascending=False)
    ranked_ev = df.screen_name.values.copy()
    ranked_evi = vmap.loc[df.index.tolist()].values.copy()
    ranked_evv = df.eigenvector.values.copy()
    ranked_df_sn = pd.DataFrame(
        dict(
            in_degree=ranked_ki,
            out_degree=ranked_ko,
            weighted_in_degree=ranked_si,
            weighted_out_degree=ranked_so,
            page_rank=ranked_pr,
            betweenness=ranked_bt,
            eigenvector=ranked_ev))
    ranked_df_sn.to_csv('centralities.ranked.screen_name.csv', index=False)
    ranked_df_id = pd.DataFrame(
        dict(
            in_degree=ranked_kii,
            out_degree=ranked_koi,
            weighted_in_degree=ranked_sii,
            weighted_out_degree=ranked_soi,
            page_rank=ranked_pri,
            betweenness=ranked_bti,
            eigenvector=ranked_evi))
    ranked_df_id.to_csv('centralities.ranked.raw_id.csv', index=False)
    ranked_df_v = pd.DataFrame(
        dict(
            in_degree=ranked_kiv,
            out_degree=ranked_kov,
            weighted_in_degree=ranked_siv,
            weighted_out_degree=ranked_sov,
            page_rank=ranked_prv,
            betweenness=ranked_btv,
            eigenvector=ranked_evv))
    ranked_df_v.to_csv('centralities.ranked.values.csv', index=False)


def distance_histogram(g):
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
    df = pd.read_csv(fn, parse_dates=['tweet_created_at'], usecols=[2, 3, 4])
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
    for created_at, from_raw_id, to_raw_id, gpf in df[[
            'from_raw_id', 'to_raw_id', 'gpf'
    ]].itertuples():
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
            logger.info('Main core at %s: k=%s, num=%s', created_at, mk, mn)
    cdf = pd.DataFrame(
        dict(
            timeline=ts,
            mcore_k=mcore_k,
            mcore_num=mcore_num,
            mcore_idx=mcore_idx,))
    cdf.to_csv(ofn, index=False)
    v_series = pd.Series(v_map)
    v_series.to_csv('vertex_map.csv')


def k_core_evolution_random_rewire(fn,
                              ofn=None,
                              freq='D',
                              model='constrained-configuration'):
    if ofn is None:
        ofn = 'k_core_evolution_{}.csv'.format(model)
    # load only necessary columns
    df = pd.read_csv(fn, parse_dates=['tweet_created_at'], usecols=[2, 3, 4])
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
    nv = []
    ne = []
    ts = []
    for created_at, from_raw_id, to_raw_id, gpf in df[[
            'from_raw_id', 'to_raw_id', 'gpf'
    ]].itertuples():
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
            g1 = g.copy()
            rejected = gt.random_rewire(g1, model=model, edge_sweep=True)
            logger.info('Number of rejected when rewiring: %s', rejected)
            ts.append(created_at)
            kcore = pd.Series(gt.kcore_decomposition(g1).a.copy())
            mcore = kcore.value_counts().sort_index(ascending=False)
            mk = mcore.index[0]
            mn = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_num.append(mn)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            logger.info(g)
            nv.append(g.num_vertices())
            ne.append(g.num_edges())
            logger.info('Main core at %s: k=%s, num=%s', created_at, mk, mn)
    cdf = pd.DataFrame(
        dict(
            timeline=ts,
            mcore_k=mcore_k,
            mcore_num=mcore_num,
            mcore_idx=mcore_idx,
            num_vertices=nv,
            num_edges=ne,))
    cdf.to_csv(ofn, index=False)


def k_core_evolution_shuffle(fn1,
                             fn2='graph.daily.csv',
                              ofn=None,
                              freq='D',
                              by='e'):
    if ofn is None:
        ofn = 'k_core_evolution_shuffle_groupby_{}.csv'.format(by)
    # load only necessary columns
    df = pd.read_csv(fn1, usecols=[3, 4])
    # remove self-loop
    df = df.loc[df.from_raw_id != df.to_raw_id]
    df = df.reindex(np.random.permutation(df.index))
    veinfo = pd.read_csv(fn2)
    if by == 'v':
        vlist = veinfo['nv'].tolist()
    elif by == 'e':
        elist = veinfo['ne'].tolist()

    v_map = dict()
    e_set = set()
    v_counter = -1
    g = gt.Graph()
    mcore_k = []
    mcore_num = []
    mcore_idx = []
    nv = []
    ne = []
    gcounter = 0
    for from_raw_id, to_raw_id in df[[ 'from_raw_id', 'to_raw_id'
                                      ]].itertuples(index=False):
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
        is_group = False
        if by == 'v':
            try:
                if g.num_vertices() == vlist[gcounter]:
                    is_group = True
                    gcounter += 1
            except IndexError:
                break
        if by == 'e':
            try:
                if g.num_edges() == elist[gcounter]:
                    is_group = True
                    gcounter += 1
            except IndexError:
                 break
        if is_group:
            kcore = pd.Series(gt.kcore_decomposition(g).a.copy())
            mcore = kcore.value_counts().sort_index(ascending=False)
            mk = mcore.index[0]
            mn = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_num.append(mn)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            nv.append(g.num_vertices())
            ne.append(g.num_edges())
            logger.info(g)
            logger.info('Main core by %s: k=%s, num=%s', by, mk, mn)
    cdf = pd.DataFrame(
        dict(
            mcore_k=mcore_k,
            mcore_num=mcore_num,
            mcore_idx=mcore_idx,
            num_vertices=nv,
            num_edges=ne,))
    cdf.to_csv(ofn, index=False)


def k_core_evolution_random_growing(fn1='retweet.201710.claim.raw.csv',
                                    fn2='graph.daily.csv',
                                    ofn=None,
                                    rewiring='configuration'
                              ):
    if ofn is None:
        cofn = 'k_core_evolution_random_growing.'
    # load only necessary columns
    g = prepare_network_from_raw(fn1)
    if rewiring is not None:
        gt.random_rewire(g, model=rewiring)
        if ofn is None:
            cofn += rewiring + '.'
    if ofn is None:
        ofn = cofn + 'csv'
    evmap = pd.read_csv(fn2)
    nelist = evmap['ne'].tolist()
    emap = pd.DataFrame(g.get_edges().copy(),
                        columns=['source', 'target', 'idx'])
    emap = emap[['source', 'target']]
    emap = emap.reindex(np.random.permutation(emap.index)
                        ).reset_index(drop=True)
    mcore_k = []
    mcore_num = []
    mcore_idx = []
    nv = []
    ne = []
    for n in nelist:
        g = gt.Graph()
        g.add_edge_list(emap.iloc[:n].values, hashed=True)
        kcore = pd.Series(gt.kcore_decomposition(g).a.copy())
        mcore = kcore.value_counts().sort_index(ascending=False)
        mk = mcore.index[0]
        mn = mcore.iloc[0]
        mcore_k.append(mk)
        mcore_num.append(mn)
        mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
        nv.append(g.num_vertices())
        ne.append(g.num_edges())
        logger.info(g)
        logger.info('Main core with num of e %s: k=%s, num=%s', n, mk, mn)
    cdf = pd.DataFrame(
        dict(
            mcore_k=mcore_k,
            mcore_num=mcore_num,
            mcore_idx=mcore_idx,
            num_vertices=nv,
            num_edges=ne,))
    cdf.to_csv(ofn, index=False)


def plot_kcore_timeline(fn='k_core_evolution.csv'):
    df = pd.read_csv(fn, parse_dates=['timeline'])
    m = df.mcore_num.groupby(df.mcore_k).max()
    m.index = df.timeline.groupby(df.mcore_k).first().values

    fig, ax = plt.subplots(figsize=(4, 3))
    ax2 = ax.twinx()
    l1, = ax2.plot(df.timeline.values, df.mcore_k, color='b')
    ax2.set_ylabel('k')
    l2, = ax.plot(m.index.values, m.values, color='r')
    ax.set_ylabel('n')
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=-30, fontsize=10)
    plt.legend([l1, l2], ['k of Main Cores', 'Size of Main Cores n'])
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
        unchanged.to_csv('k_core.daily.intersection.csv')
        unions.to_csv('k_core.daily.union.csv')
        logger.info('Number of unchanged is %s', len(unchanged))
        logger.info('Number of union is %s', len(unions))


def main_core(df):
    w = df.groupby(['from_raw_id', 'to_raw_id']).size().\
        rename('weight').reset_index(drop=False)
    g = gt.Graph()
    v_raw_ids = g.add_edge_list(
        w[['from_raw_id', 'to_raw_id']].values, hashed=True).a.copy()
    kcores = gt.kcore_decomposition(g).a.copy()
    s = pd.Series(kcores)
    return v_raw_ids[s.loc[s == s.max()].index]


def rolling_k_core(fn='retweet.201710.claim.raw.csv'):
    df = pd.read_csv(fn, parse_dates=['tweet_created_at'], usecols=[2, 3, 4])
    df = df.set_index('tweet_created_at')
    # remove self-loop
    df = df.loc[df.from_raw_id != df.to_raw_id]
    df1 = df.loc[:'2016-11-07']
    df2 = df.loc['2016-11-08':'2017-04-07']
    df3 = df.loc['2017-04-08':]
    logger.info('Dataset 1: rows %s, days %s',
                len(df1), df1.index.max() - df1.index.min())
    logger.info('Dataset 2: rows %s, days %s',
                len(df2), df2.index.max() - df2.index.min())
    logger.info('Dataset 3: rows %s, days %s',
                len(df3), df3.index.max() - df3.index.min())
    s1 = set(main_core(df1))
    s2 = set(main_core(df2))
    s3 = set(main_core(df3))
    unchanged = pd.DataFrame(list(s1 & s2 & s3), columns=['user_raw_id'])
    unions = pd.DataFrame(list(s1 | s2 | s3), columns=['user_raw_id'])
    logger.info('Intersection: %s', len(unchanged))
    logger.info('Union: %s', len(unions))
    unchanged.to_csv('k_core.3.intersection.csv', index=False)
    unions.to_csv('k_core.3.union.csv', index=False)


def rearrange_ranked_centralities(fn='centralities.20.raw.csv'):
    df = pd.read_csv(fn)
    data = []
    for c in df.columns:
        data += [(i, c, v) for i, v in df[c].iteritems()]
    ndf = pd.DataFrame(data, columns=['rank', 'centrality', 'screen_name'])
    ndf.to_csv('centralities.20.csv', index=False)


def vid2sn(vids, fn='vmap.csv', vmap=None):
    if vmap is None:
        vmap = pd.read_csv('vmap.csv')
    ivmap = vmap.set_index('vid').screen_name
    return ivmap.loc[vids].tolist()
