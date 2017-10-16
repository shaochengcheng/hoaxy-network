import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)
C1 = '#1F78B4'
C2 = '#FF7F00'
FIGSIZE = (4, 3)


def bot_centrality_rank(top=1000,
                        fn1='ubs.csv',
                        fn2='centralities.ranked.raw_id.csv',
                        fn3='centralities.ranked.values.csv'):
    if top > 1000:
        raise ValueError('Top should not larger than 1000!')
    df1 = pd.read_csv(fn1)
    bmap = df1.set_index('user_raw_id').bot_score_en
    df2 = pd.read_csv(fn2)
    df3 = pd.read_csv(fn3)
    df2 = df2.iloc[:top]
    df3 = df3.iloc[:top]
    correlations = []
    for c in df3.columns:
        bs = bmap.loc[df2[c].values]
        df = pd.DataFrame(
            dict(centrality=df3[c].values.copy(), bot_score=bs.values.copy()))
        df = df.loc[df.bot_score.notnull()]
        a1 = df.centrality.values
        a2 = df.bot_score.values
        rho, rhop = spearmanr(a1, a2)
        tau, taup = kendalltau(a1, a2)
        correlations.append((c, 'spearmanr', rho, rhop))
        correlations.append((c, 'kendalltau', tau, taup))
    df = pd.DataFrame(correlations)
    df.to_csv('bot_centrality_correlation.{}.csv'.format(top), index=False)
    return df


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
    return df


def centrality_vs_rand(fn1='ubs.csv',
                       fn2='retweet.1108.random.5000.csv',
                       fn3='centralities.ranked.raw_id.csv',
                       ofn='bots-of-users.pdf',
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


def mcore_vs_rand(fn1='ubs.csv',
                  fn2='retweet.1108.random.5000.csv',
                  fn3='retweet.1108.claim.kcore.raw.csv',
                  ofn='bots-of-users.pdf',
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


def mcore_vs_centrality(fn1='ubs.csv',
                        fn2='retweet.1108.claim.kcore.raw.csv',
                        fn3='centralities.ranked.raw_id.csv',
                        ofn='bots-of-users.pdf',
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
