import logging

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator
from matplotlib import gridspec

from .data_process import get_data_file, get_out_file, ccdf
from .data_process import nplog

logger = logging.getLogger(__name__)


def build_network(fn):
    fn = get_data_file(fn)
    df = pd.read_csv(fn)
    from_names = df[['from_raw_id', 'from_screen_name']].copy()
    from_names.columns = ['raw_id', 'screen_name']
    to_names = df[['to_raw_id', 'to_screen_name']].copy()
    to_names.columns = ['raw_id', 'screen_name']
    names = pd.concat([from_names, to_names], ignore_index=True)
    names = names.drop_duplicates()
    names = names.set_index('raw_id')['screen_name']
    g = nx.from_pandas_dataframe(
        df,
        source='from_raw_id',
        target='to_raw_id',
        edge_attr='weight',
        create_using=nx.DiGraph())
    nx.set_node_attributes(g, name='screen_name', values=names.to_dict())
    return g


def all_degrees(g):
    return dict(
        ki=pd.Series(g.in_degree(), name='ki'),
        ko=pd.Series(g.out_degree(), name='ko'),
        si=pd.Series(g.in_degree(weight='weight'), name='si'),
        so=pd.Series(g.out_degree(weight='weight'), name='so'))


def deg_hub_stat(g, deg, fn, top=10):
    fn = get_data_file(fn)
    names = pd.Series(nx.get_node_attributes(g, name='screen_name'))
    dfs = []
    for k, v in deg.items():
        deg[k] = v.sort_values(ascending=False)
        hub = deg[k].iloc[:top].copy()
        hub_sn = names.loc[hub.index]
        hub_df = pd.concat([hub, hub_sn], axis=1)
        hub_df = hub_df.reset_index(drop=False)
        k_raw_id = k + '_raw_id'
        k_value = k + '_value'
        k_screen_name = k + '_screen_name'
        hub_df.columns = [k_raw_id, k_value, k_screen_name]
        dfs.append(hub_df)
    df = pd.concat(dfs, axis=1)
    df.to_csv(fn, index=False)
    print(df)


def plot_deg_dist(deg1, deg2, figsize=(8, 6)):
    ccdf_deg1 = dict()
    ccdf_deg2 = dict()
    for k, v in deg1.items():
        ccdf_deg1[k] = ccdf(v)

    for k, v in deg2.items():
        ccdf_deg2[k] = ccdf(v)

    titles = (('ki', 'In Degree'), ('ko', 'Out Degree'),
              ('si', 'Weighted In Degree'), ('so', 'Weighted Out Degree'))

    fig, axarr = plt.subplots(2, 2, figsize=figsize)
    axarr = axarr.flatten()
    for i, tt in enumerate(titles):
        k, t = tt
        axarr[i].set_title(t)
        axarr[i].loglog(
            ccdf_deg1[k].index + 1, ccdf_deg1[k].values, label='Claim')
        axarr[i].loglog(
            ccdf_deg2[k].index + 1, ccdf_deg2[k].values, label='Fact Checking')
    fig.tight_layout()


def prepare_deg_heatmap(df, base=2):
    c = df.columns
    X = df[c[0]].values + 1
    Y = df[c[1]].values + 1
    ximax = int(np.ceil(nplog(X.max(), base)))
    yimax = int(np.ceil(nplog(Y.max(), base)))
    xbins = [np.power(base, i) for i in range(ximax + 1)]
    ybins = [np.power(base, i) for i in range(yimax + 1)]
    H, xedges, yedges = np.histogram2d(X, Y, bins=[xbins, ybins])
    return H, xedges, yedges


def ax_deg_heatmap(ax, X, Y, H, vmin, vmax):
    # heatmap
    return ax.pcolormesh(
        X,
        Y,
        H.T,
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap='gnuplot2_r')


def plot_deg_heatmap(deg1, deg2, base=2, figsize=(6, 5)):
    fig = plt.figure(figsize=figsize)
    fs = 9
    gs = gridspec.GridSpec(
        2,
        3,
        wspace=0.15,
        hspace=0.25,
        width_ratios=[8, 8, 1],
        height_ratios=[0.5, 0.5])
    axarr = []
    axarr.append(fig.add_subplot(gs[0, 0]))
    axarr.append(fig.add_subplot(gs[0, 1]))
    axarr.append(fig.add_subplot(gs[1, 0]))
    axarr.append(fig.add_subplot(gs[1, 1]))
    axarr.append(fig.add_subplot(gs[:, 2]))
    axarr[0].set_title('Degree (Claim)', fontsize=fs)
    axarr[1].set_title('Weigthed Degree (Claim)', fontsize=fs)
    axarr[2].set_title('Degree (Fact Checking)', fontsize=fs)
    axarr[3].set_title('Weigthed Degree (Fact Checking)', fontsize=fs)
    df = []
    df.append(pd.concat((deg1['ki'], deg1['ko']), axis=1))
    df.append(pd.concat((deg1['si'], deg1['so']), axis=1))
    df.append(pd.concat((deg2['ki'], deg2['ko']), axis=1))
    df.append(pd.concat((deg2['si'], deg2['so']), axis=1))
    X = []
    Y = []
    XE = []
    YE = []
    H = []
    for d in df:
        c = d.columns
        X.append(d[c[0]].values + 1)
        Y.append(d[c[1]].values + 1)
    ximax = int(np.ceil(nplog(max(x.max() for x in X), base)))
    yimax = int(np.ceil(nplog(max(y.max() for y in Y), base)))
    xbins = [np.power(base, i) for i in range(ximax + 1)]
    ybins = [np.power(base, i) for i in range(yimax + 1)]
    for i in range(4):
        h, xedges, yedges = np.histogram2d(X[i], Y[i], bins=[xbins, ybins])
        H.append(h)
        XE.append(xedges)
        YE.append(yedges)
    vmin = min(h.min() for h in H) + 1
    vmax = max(h.max() for h in H)
    for i in range(4):
        xm, ym = np.meshgrid(XE[i], YE[i])
        im = axarr[i].pcolormesh(
            xm,
            ym,
            H[i].T,
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap='gnuplot2_r')
        axarr[i].set_xscale('log')
        axarr[i].set_yscale('log')
    axarr[0].set_xticklabels([])
    axarr[0].set_ylabel('Out')
    axarr[1].set_xticklabels([])
    axarr[1].set_yticklabels([])
    axarr[2].set_ylabel('Out')
    axarr[2].set_xlabel('In')
    axarr[3].set_yticklabels([])
    axarr[3].set_xlabel('In')
    axarr[4].axis('off')
    # axarr[3].xaxis.set_major_locator(LogLocator(base=10))
    # axarr[3].tick_params(axis='x', which='minor', length=4, color='r')
    plt.colorbar(im, ax=axarr[4], orientation='vertical', fraction=0.8)
    # fig.tight_layout()


def mention_deg_dist(fn1='mention.20170921.fn.csv',
                     fn2='mention.20170921.fc.csv',
                     ofn='mention-degree-dist.pdf',
                     ofn1='mention.20170921.hub.fn.csv',
                     ofn2='mention.20170921.hub.fc.csv',
                     top=10,
                     figsize=(8, 6)):
    ofn = get_out_file(ofn)
    g1 = build_network(fn1)
    g2 = build_network(fn2)
    deg1 = all_degrees(g1)
    deg2 = all_degrees(g2)
    print('Mention of fake news:\n')
    deg_hub_stat(g1, deg1, ofn1, top=top)
    print('Mention of fact checking\n')
    deg_hub_stat(g2, deg2, ofn2, top=top)
    plot_deg_dist(deg1, deg2, figsize)
    plt.savefig(ofn)


def retweet_deg_dist(fn1='retweet.20170921.fn.csv',
                     fn2='retweet.20170921.fc.csv',
                     ofn='retweet-degree-dist.pdf',
                     ofn1='retweet.20170921.hub.fn.csv',
                     ofn2='retweet.20170921.hub.fc.csv',
                     top=10,
                     figsize=(8, 6)):
    ofn = get_out_file(ofn)
    g1 = build_network(fn1)
    g2 = build_network(fn2)
    deg1 = all_degrees(g1)
    deg2 = all_degrees(g2)
    print('Retweet of fake news:\n')
    deg_hub_stat(g1, deg1, ofn1, top=top)
    print('Retweet of fact checking\n')
    deg_hub_stat(g2, deg2, ofn2, top=top)
    plot_deg_dist(deg1, deg2, figsize)
    plt.savefig(ofn)


def mention_deg_heatmap(fn1='mention.20170921.fn.csv',
                        fn2='mention.20170921.fc.csv',
                        ofn='mention-degree-heatmap.pdf',
                        base=2,
                        figsize=(6, 5)):
    ofn = get_out_file(ofn)
    g1 = build_network(fn1)
    g2 = build_network(fn2)
    deg1 = all_degrees(g1)
    deg2 = all_degrees(g2)
    plot_deg_heatmap(deg1, deg2, base=base, figsize=figsize)
    plt.savefig(ofn)


def retweet_deg_heatmap(fn1='retweet.20170921.fn.csv',
                        fn2='retweet.20170921.fc.csv',
                        ofn='retweet-degree-heatmap.pdf',
                        base=2,
                        figsize=(6, 5)):
    ofn = get_out_file(ofn)
    g1 = build_network(fn1)
    g2 = build_network(fn2)
    deg1 = all_degrees(g1)
    deg2 = all_degrees(g2)
    plot_deg_heatmap(deg1, deg2, base=base, figsize=figsize)
    plt.savefig(ofn)
