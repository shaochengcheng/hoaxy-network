import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu

import logging

logger = logging.getLogger(__name__)
C1 = '#1F78B4'
C2 = '#FF7F00'
FIGSIZE = (4, 3)


def kcore_timeline(fns=[
        'kcore.growing.csv', 'kcore.growing.shuffle.csv',
        'kcore.growing.weighted-shuffle.csv', 'kcore.growing.ba.csv'
],
                   labels=['Shuffle', 'Weighted Shuffle', 'BA']):
    df = pd.read_csv(fns[0], parse_dates=['timeline'])
    df = df[['timeline', 'mcore_k', 'mcore_s']]
    df = df.rename(columns=dict(
        timeline='Timeline', mcore_k='K', mcore_s='Size'))
    secondary_c = ['K']
    for i, fn in enumerate(fns[1:]):
        df2 = pd.read_csv(fns[1])
        df2 = df2[['mcore_k', 'mcore_s']]
        kname = 'K, ' + labels[i]
        sname = 'Size, ' + labels[i]
        df2.columns = [kname, sname]
        secondary_c.append(kname)
        df = pd.merge(df, df2, left_index=True, right_index=True)
    ldf = df.groupby('K').last()
    ldf = ldf.set_index('Timeline')
    fig, ax = plt.subplots()
    ldf.plot(ax=ax, secondary_y=secondary_c)
    plt.tight_layout()


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


def mcore(fn, bidx=None):
    df = pd.read_csv(fn)
    df[['mcore_k', 'mcore_s']].plot(secondary_y=['mcore_k'])
    if bidx is None:
        return
    df['mcore_idx'] = df.mcore_idx.apply(eval).apply(set)
    s0 = df.iloc[bidx].mcore_idx
    print('Staring number: %s' % len(s0))
    for idx, s in df.mcore_idx.iloc[bidx + 1:].iteritems():
        s0 &= s
    print('Unchanged number: %s' % len(s0))


def hoaxy_usage(fn='hoaxy.usage.csv', ofn=None, top=2, logy=False):
    """The usage of hoaxy frontend."""
    if ofn is None:
        ofn = 'hoaxy-usage.pdf'
    df = pd.read_csv(fn, parse_dates=['timeline'])
    df = df.set_index('timeline')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    df.counts.plot(ax=ax, logy=logy, color=C1)
    counter = 0
    for ts, v, tlabels in df.loc[df.tlabels.notnull()].itertuples():
        label = '\n'.join(eval(tlabels))
        if counter == 1:
            v += 500
        else:
            v += 50
        ax.text(
            ts,
            v,
            label,
            horizontalalignment='center',
            fontsize=9,
            verticalalignment='bottom')
        counter += 1
    df.loc[df.tlabels.notnull()].counts.plot(
        ax=ax, linestyle='None', marker='s', markersize=4, color=C2, alpha=0.6)
    ax.set_xlim(['2016-12-15', '2017-04-10'])
    ax.set_ylim([1e1, 1e5])
    ax.set_xlabel('Timeline')
    ax.set_ylabel('Daily Query Volume')
    plt.tight_layout()
    plt.savefig(ofn)


def mcore_centrality_overlapping(tops=np.array(list(range(100, 2001, 100))),
                                 fn1='centralities.ranked.raw_id.csv',
                                 fn2='retweet.1108.claim.kcore.raw.csv'):
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    s0 = set(df2.loc[df2.kcore == df2.kcore.max()].raw_id.values)
    r = []
    for top in tops:
        df = df1.iloc[:top]
        jaccard = []
        for c in df.columns:
            s1 = set(df[c].values)
            jaccard.append(len(s1 & s0) / len(s0))
            # jaccard.append(len(s1 & s0)/len(s1 | s0))
        r.append(jaccard)
    df = pd.DataFrame(r, columns=df1.columns, index=tops)
    df.plot()
    plt.savefig('mcore_centrality_overlapping.pdf')


def bot_centrality_vs_rand(fn1='ubs.csv',
                           fn2='retweet.1108.random.5000.csv',
                           fn3='centralities.ranked.raw_id.csv',
                           ofn='bots-centrality-vs-rand.pdf',
                           sample_size=None,
                           nbins=20,
                           normed=True,
                           figsize=FIGSIZE):
    # pdb.set_trace()
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df3 = pd.read_csv(fn3)

    bmap = df1.set_index('user_raw_id').bot_score_en
    s1 = bmap.loc[np.unique(df3.iloc[:1000].values.flatten())]
    s2 = bmap.loc[df2.raw_id.values]
    a1 = s1.loc[s1.notnull()].values
    a2 = s2.loc[s2.notnull()].values
    mu1 = np.mean(a1)
    sigma1 = np.std(a1, ddof=1)
    mu2 = np.mean(a2)
    sigma2 = np.std(a2, ddof=1)

    logger.info('Number of Non-nan values: len(centrality)=%s, len(rand)=%s',
                len(a1), len(a2))
    logger.info('mu1=%s, mu2=%s', mu1, mu2)
    logger.info('sigma1=%s, sigma2=%s', sigma1, sigma2)

    logger.info('Welch\'s t-test: %s',
                ttest_ind(a1, a2, equal_var=False, nan_policy='raise'))
    logger.info('Kolmogorov-Smirnov test: %s', ks_2samp(a1, a2))
    logger.info('Mann Whitney U test: %s',
                mannwhitneyu(a1, a2, alternative='two-sided'))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, nbins + 1)
    if normed is False:
        w1 = np.ones_like(a1) / len(a1)
        w2 = np.ones_like(a2) / len(a2)
    else:
        w1 = None
        w2 = None
    ax.set_xlim([0, 1])
    ax.hist(
        a1,
        bins,
        weights=w1,
        normed=normed,
        alpha=0.5,
        label='High Centrality',
        color=C1)
    ax.hist(
        a2,
        bins,
        weights=w2,
        normed=normed,
        alpha=0.5,
        label='Random Sample',
        color=C2)
    plt.legend(loc='upper right', fontsize='small')
    ax.set_xlabel('Bot Score')
    if normed is True:
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel('$proportion$')
    plt.tight_layout()
    plt.savefig(ofn)


def bot_mcore_vs_rand(fn1='ubs.csv',
                      fn2='retweet.1108.random.5000.csv',
                      fn3='retweet.1108.claim.kcore.raw.csv',
                      ofn='bots-mcore-vs-rand.pdf',
                      sample_size=None,
                      nbins=20,
                      normed=True,
                      figsize=FIGSIZE):
    # pdb.set_trace()
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df3 = pd.read_csv(fn3)

    bmap = df1.set_index('user_raw_id').bot_score_en
    s1 = bmap.loc[df3.loc[df3.kcore == df3.kcore.max()].raw_id.values]
    s2 = bmap.loc[df2.raw_id.values]
    a1 = s1.loc[s1.notnull()].values
    a2 = s2.loc[s2.notnull()].values
    mu1 = np.mean(a1)
    sigma1 = np.std(a1, ddof=1)
    mu2 = np.mean(a2)
    sigma2 = np.std(a2, ddof=1)

    logger.info('Number of Non-nan values: len(mcores)=%s, len(rand)=%s',
                len(a1), len(a2))
    logger.info('mu1=%s, mu2=%s', mu1, mu2)
    logger.info('sigma1=%s, sigma2=%s', sigma1, sigma2)

    logger.info('Welch\'s t-test: %s',
                ttest_ind(a1, a2, equal_var=False, nan_policy='raise'))
    logger.info('Kolmogorov-Smirnov test: %s', ks_2samp(a1, a2))
    logger.info('Mann Whitney U test: %s',
                mannwhitneyu(a1, a2, alternative='two-sided'))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, nbins + 1)
    if normed is False:
        w1 = np.ones_like(a1) / len(a1)
        w2 = np.ones_like(a2) / len(a2)
    else:
        w1 = None
        w2 = None
    ax.set_xlim([0, 1])
    ax.hist(
        a1,
        bins,
        weights=w1,
        normed=normed,
        alpha=0.5,
        label='Main cores',
        color=C1)
    ax.hist(
        a2,
        bins,
        weights=w2,
        normed=normed,
        alpha=0.5,
        label='Random Sample',
        color=C2)
    plt.legend(loc='upper right', fontsize='small')
    ax.set_xlabel('Bot Score')
    if normed is True:
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel('$proportion$')
    plt.tight_layout()
    plt.savefig(ofn)


def bot_mcore_vs_centrality(fn1='ubs.csv',
                            fn2='retweet.1108.claim.kcore.raw.csv',
                            fn3='centralities.ranked.raw_id.csv',
                            ofn='bots-mcore-vs-centrality.pdf',
                            sample_size=None,
                            nbins=20,
                            normed=True,
                            figsize=FIGSIZE):
    # pdb.set_trace()
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df3 = pd.read_csv(fn3)

    bmap = df1.set_index('user_raw_id').bot_score_en
    s1 = bmap.loc[df2.loc[df2.kcore == df2.kcore.max()].raw_id.values]
    s2 = bmap.loc[np.unique(df3.iloc[:1000].values.flatten())]
    a1 = s1.loc[s1.notnull()].values
    a2 = s2.loc[s2.notnull()].values
    mu1 = np.mean(a1)
    sigma1 = np.std(a1, ddof=1)
    mu2 = np.mean(a2)
    sigma2 = np.std(a2, ddof=1)

    logger.info('Number of Non-nan values: len(mcores)=%s, len(centrality)=%s',
                len(a1), len(a2))
    logger.info('mu1=%s, mu2=%s', mu1, mu2)
    logger.info('sigma1=%s, sigma2=%s', sigma1, sigma2)

    logger.info('Welch\'s t-test: %s',
                ttest_ind(a1, a2, equal_var=False, nan_policy='raise'))
    logger.info('Kolmogorov-Smirnov test: %s', ks_2samp(a1, a2))
    logger.info('Mann Whitney U test: %s',
                mannwhitneyu(a1, a2, alternative='two-sided'))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, nbins + 1)
    if normed is False:
        w1 = np.ones_like(a1) / len(a1)
        w2 = np.ones_like(a2) / len(a2)
    else:
        w1 = None
        w2 = None
    ax.set_xlim([0, 1])
    ax.hist(
        a1,
        bins,
        weights=w1,
        normed=normed,
        alpha=0.5,
        label='Main cores',
        color=C1)
    ax.hist(
        a2,
        bins,
        weights=w2,
        normed=normed,
        alpha=0.5,
        label='Centrality',
        color=C2)
    plt.legend(loc='upper right', fontsize='small')
    ax.set_xlabel('Bot Score')
    if normed is True:
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel('$proportion$')
    plt.tight_layout()
    plt.savefig(ofn)
