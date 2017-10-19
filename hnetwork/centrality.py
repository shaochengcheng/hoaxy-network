import logging

try:
    import graph_tool.all as gt
except ImportError:
    print('Some function Require graph_tool!')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def weight_edge_list(fn, ofn):
    """Get the weight of edges and save to file.

    Parameters
    ----------
    fn: string
        input csv file name, contains the retweets.
    ofn: string
        output csv file name, contains the weighted edge list.
    """
    df = pd.read_csv(fn)
    df = df.loc[df.from_raw_id != df.to_raw_id]
    weight = df.groupby(['from_raw_id', 'to_raw_id']).size().rename('weight')
    weight.to_csv(ofn, index=True)


def load_graph(fn):
    """Load graph_tool.Graph from weighted edge list."""
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
    """Load graph_tool.Graph from raw timestamped retweets."""
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
    """Use graph_tool to calculate 7 centralities."""
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


def rank_centralities(fn1='centralities.raw.csv',
                      fn2='retweet.1108.claim.vmap.csv'):
    """Rank centralities."""
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
    """Use graph_tool to get the Distance Histogram."""
    counts, bins = gt.distance_histogram(g)
    df = pd.DataFrame(dict(counts=counts))
    df.to_csv('distance_histogram.csv', index=False)


def v_percolate(g, vertices, ofn):
    """Vertex percolation"""
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


def kcore_growing(fn, ofn=None, freq='D'):
    """The growing of kcores for raw data."""
    if ofn is None:
        ofn = 'kcore_growing.csv'
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
    mcore_s = []
    mcore_idx = []
    vnum = []
    enum = []
    largest_component_vnum = []
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
            ms = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_s.append(ms)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            lcv = gt.label_largest_component(g, directed=False)
            vnum.append(g.num_vertices())
            enum.append(g.num_edges())
            largest_component_vnum.append(lcv.a.sum())
            logger.info(g)
            logger.info('Main core at %s: k=%s, num=%s', created_at, mk, ms)
    cdf = pd.DataFrame(
        dict(
            timeline=ts,
            mcore_k=mcore_k,
            mcore_s=mcore_s,
            mcore_idx=mcore_idx,
            vnum=vnum,
            enum=enum,
            largest_commponent_vnum=largest_component_vnum
        ))
    cdf.to_csv(ofn, index=False)
    v_series = pd.Series(v_map)
    v_series.name = 'raw_id'
    v_series.index.name = 'v_idx'
    v_series.to_csv('vertex_map.csv', index=True, header=True)


def kcore_growing_daily_rewiring(fn,
                              ofn=None,
                              freq='D',
                              model='constrained-configuration'):
    """The growing of kcore by rewiring daily."""
    if ofn is None:
        ofn = 'kcore.growing.daily-rewiring.{}.csv'.format(model)
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
    mcore_s = []
    mcore_idx = []
    vnum = []
    enum = []
    largest_component_vnum = []
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
            ms = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_s.append(ms)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            lcv = gt.label_largest_component(g1, directed=False)
            vnum.append(g1.num_vertices())
            enum.append(g1.num_edges())
            largest_component_vnum.append(lcv.a.sum())
            logger.info(g1)
            logger.info('Main core at %s: k=%s, num=%s', created_at, mk, ms)
    cdf = pd.DataFrame(
        dict(
            timeline=ts,
            mcore_k=mcore_k,
            mcore_s=mcore_s,
            mcore_idx=mcore_idx,
            vnum=vnum,
            enum=enum,
            largest_commponent_vnum=largest_component_vnum
        ))
    cdf.to_csv(ofn, index=False)


def kcore_growing_weighted_shuffle(fn1,
                                fn2='graph.daily.csv',
                             ofn=None,
                             freq='D'
                             ):
    """The growing of kcore by shuffling the retweet list."""
    if ofn is None:
        ofn = 'kcore.growing.weighted-shuffle.csv'
    # load only necessary columns
    df = pd.read_csv(fn1, usecols=[3, 4])
    # remove self-loop
    df = df.loc[df.from_raw_id != df.to_raw_id]
    df = df.reindex(np.random.permutation(df.index))
    evmap = pd.read_csv(fn2)
    enum_list = evmap['enum'].tolist()
    v_map = dict()
    v_counter = -1
    e_set = set()
    gp_counter = 0
    g = gt.Graph()
    mcore_k = []
    mcore_s = []
    mcore_idx = []
    vnum = []
    enum = []
    largest_component_vnum = []
    ts = []
    g = gt.Graph()
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
        if g.num_edges() >= enum_list[gp_counter]:
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
            ms = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_s.append(ms)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            lcv = gt.label_largest_component(g, directed=False)
            vnum.append(g.num_vertices())
            enum.append(g.num_edges())
            largest_component_vnum.append(lcv.a.sum())
            logger.info(g)
            logger.info('gp counter: %s', gp_counter)
            logger.info('Main core at enum=%s: k=%s, num=%s',
                        g.num_edges(), mk, ms)
            gp_counter += 1
            if gp_counter > len(enum_list):
                break
    cdf = pd.DataFrame(
        dict(
            mcore_k=mcore_k,
            mcore_s=mcore_s,
            mcore_idx=mcore_idx,
            vnum=vnum,
            enum=enum,
            largest_commponent_vnum=largest_component_vnum
        ))
    cdf.to_csv(ofn, index=False)


def kcore_growing_shuffle(fn1='retweet.201710.claim.raw.csv',
                          fn2='graph.daily.csv',
                          ofn=None,
                          rewiring=None
                          ):
    """The growing of kcore by shuffling the edge list."""
    if ofn is None:
        ofn = 'kcore.growing.shuffle'
        if rewiring:
            ofn += '.' + rewiring
        ofn += '.csv'
    g = prepare_network_from_raw(fn1)
    if rewiring is not None:
        gt.random_rewire(g, model=rewiring)
    evmap = pd.read_csv(fn2)
    enum_list = evmap['enum'].tolist()
    emap = pd.DataFrame(g.get_edges().copy(),
                        columns=['source', 'target', 'idx'])
    emap = emap[['source', 'target']]
    emap = emap.reindex(np.random.permutation(emap.index)
                        ).reset_index(drop=True)
    v_map = dict()
    v_counter = -1
    gp_counter = 0
    g = gt.Graph()
    mcore_k = []
    mcore_s = []
    mcore_idx = []
    vnum = []
    enum = []
    largest_component_vnum = []
    g = gt.Graph()
    for i, s, t in emap.itertuples():
        if s not in v_map:
            v_counter += 1
            v_map[s] = v_counter
        if t not in v_map:
            v_counter += 1
            v_map[t] = v_counter
        source = v_map.get(s)
        target = v_map.get(t)
        g.add_edge(source, target, add_missing=True)
        if g.num_edges() >= enum_list[gp_counter]:
            kcore = pd.Series(gt.kcore_decomposition(g).a.copy())
            mcore = kcore.value_counts().sort_index(ascending=False)
            mk = mcore.index[0]
            ms = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_s.append(ms)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            lcv = gt.label_largest_component(g, directed=False)
            vnum.append(g.num_vertices())
            enum.append(g.num_edges())
            largest_component_vnum.append(lcv.a.sum())
            logger.info(g)
            logger.info('gp counter: %s', gp_counter)
            logger.info('Main core at enum=%s: k=%s, num=%s',
                        g.num_edges(), mk, ms)
            gp_counter += 1
    cdf = pd.DataFrame(
        dict(
            mcore_k=mcore_k,
            mcore_s=mcore_s,
            mcore_idx=mcore_idx,
            vnum=vnum,
            enum=enum,
            largest_commponent_vnum=largest_component_vnum
        ))
    cdf.to_csv(ofn, index=False)


def kcore_growing_ba(fn1='ba.gml',
                          fn2='graph.daily.csv',
                          ofn=None,
                          ):
    """The growing of kcore for a BA model."""
    if ofn is None:
        ofn = 'kcore.growing.ba.csv'
    g = gt.load_graph(fn1)
    evmap = pd.read_csv(fn2)
    vnum_list = evmap['vnum'].tolist()
    emap = pd.DataFrame(g.get_edges().copy(),
                        columns=['source', 'target', 'idx'])
    emap = emap[['source', 'target']]
    v_map = dict()
    v_counter = -1
    gp_counter = 0
    g = gt.Graph()
    mcore_k = []
    mcore_s = []
    mcore_idx = []
    vnum = []
    enum = []
    largest_component_vnum = []
    g = gt.Graph()
    for i, s, t in emap.itertuples():
        if s not in v_map:
            v_counter += 1
            v_map[s] = v_counter
        if t not in v_map:
            v_counter += 1
            v_map[t] = v_counter
        source = v_map.get(s)
        target = v_map.get(t)
        g.add_edge(source, target, add_missing=True)
        if g.num_vertices() >= vnum_list[gp_counter]:
            kcore = pd.Series(gt.kcore_decomposition(g).a.copy())
            mcore = kcore.value_counts().sort_index(ascending=False)
            mk = mcore.index[0]
            ms = mcore.iloc[0]
            mcore_k.append(mk)
            mcore_s.append(ms)
            mcore_idx.append(kcore.loc[kcore == mk].index.tolist())
            lcv = gt.label_largest_component(g, directed=False)
            vnum.append(g.num_vertices())
            enum.append(g.num_edges())
            largest_component_vnum.append(lcv.a.sum())
            logger.info(g)
            logger.info('gp counter: %s', gp_counter)
            logger.info('Main core at vnum=%s: k=%s, num=%s',
                        g.num_vertices(), mk, ms)
            gp_counter += 1
            try:
                vnum_list[gp_counter]
            except IndexError:
                break
    cdf = pd.DataFrame(
        dict(
            mcore_k=mcore_k,
            mcore_s=mcore_s,
            mcore_idx=mcore_idx,
            vnum=vnum,
            enum=enum,
            largest_commponent_vnum=largest_component_vnum
        ))
    cdf.to_csv(ofn, index=False)

