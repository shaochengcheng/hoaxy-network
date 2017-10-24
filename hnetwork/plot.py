import pandas as pd
import numpy as np
import matplotlib as mpl
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


def hoaxy_usage(fn='hoaxy.usage.csv', ofn=None, top=2, logy=True):
    """The usage of hoaxy frontend."""
    if ofn is None:
        ofn = 'hoaxy-usage.pdf'
    df = pd.read_csv(fn, parse_dates=['timeline'])
    df = df.set_index('timeline')
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df.counts.plot(ax=ax, logy=logy, color=C1)
    counter = 0
    lfontsize = 8
    for ts, v, tlabels in df.loc[df.tlabels.notnull()].itertuples():
        label = '\n'.join(eval(tlabels))
        if counter == 0:
            v = v + 4000
        if counter == 1:
            ax.annotate(
                label,
                xy=(ts, v + 50),
                xytext=(ts + pd.Timedelta('8 days'), 10000),
                fontsize=lfontsize,
                horizontalalignment='left',
                verticalalignment='bottom',
                arrowprops=dict(
                    facecolor='black',
                    width=0.8,
                    headlength=6,
                    headwidth=4,
                    alpha=0.4,
                    shrink=0.02))
        elif counter == 3:
            ax.annotate(
                label,
                xy=(ts, v + 50),
                xytext=(ts + pd.Timedelta('13 days'), 8500),
                fontsize=lfontsize,
                horizontalalignment='left',
                verticalalignment='bottom',
                arrowprops=dict(
                    facecolor='black',
                    width=0.8,
                    headlength=6,
                    headwidth=4,
                    alpha=0.4,
                    shrink=0.02))
        elif counter == 5:
            ax.annotate(
                label,
                xy=(ts, v + 50),
                xytext=(ts + pd.Timedelta('6 days'), 3200),
                fontsize=lfontsize,
                horizontalalignment='left',
                verticalalignment='bottom',
                arrowprops=dict(
                    facecolor='black',
                    width=0.8,
                    headlength=6,
                    headwidth=4,
                    alpha=0.4,
                    shrink=0.02))
        else:
            ax.text(
                ts - pd.Timedelta('2 days'),
                v + 200,
                label,
                horizontalalignment='left',
                fontsize=lfontsize,
                verticalalignment='bottom')
        counter += 1
    df.loc[df.tlabels.notnull()].counts.plot(
        ax=ax, linestyle='None', marker='s', markersize=4, color=C2, alpha=0.6)
    ax.fill_between(
        df.index.to_pydatetime(), 0, df.counts.values, facecolor='#E6F4FA')
    ax.set_xlim(['2016-12-16', '2017-04-26'])
    ax.set_ylim([1e1, 4e4])
    ax.set_xlabel('')
    ax.set_ylabel('Daily Query Volume')
    ax.tick_params(
        axis='x', which='minor', bottom='off', top='off', labelbottom='off')
    plt.tight_layout()
    plt.savefig(ofn)


def mcore_centrality_overlapping1(tops=np.array(list(range(100, 2001, 100))),
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


def rank_position_mcore_centrality_box(fn1='retweet.1108.claim.kcore.raw.csv',
                                       fn2='centralities.ranked.raw_id.csv'):
    """The position of main cores located at different centralities."""
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df2['rank_id'] = df2.index.values + 1
    df1 = df1.loc[df1.kcore == df1.kcore.max()]
    data = []
    for c in df2.columns[:-1]:
        ranks = pd.merge(
            df1, df2, how='inner', left_on='raw_id', right_on=c).rank_id.values
        data.append(ranks)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(
        data,
        showfliers=False,
        whis=[2.5, 97.5],
        labels=df2.columns[:-1],
        vert=False,)
    ax.set_xscale('log')
    ax.set_xlim([1e0, 1e6])
    ax.set_xlabel('Ranking')
    plt.tight_layout()
    plt.savefig('rank-position-mcore-centrality-box.pdf')


def rank_position_mcore_centrality_violin(
        fn1='retweet.1108.claim.kcore.raw.csv',
        fn2='centralities.ranked.raw_id.csv'):
    """The position of main cores located at different centralities."""
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df2['rank_id'] = df2.index.values + 10
    df1 = df1.loc[df1.kcore == df1.kcore.max()]
    data = []
    for c in df2.columns[:-1]:
        ranks = pd.merge(
            df1, df2, how='inner', left_on='raw_id', right_on=c).rank_id.values
        ranks = np.log10(ranks)
        data.append(ranks)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.violinplot(data, vert=False)
    plt.yticks([1, 2, 3, 4, 5, 6, 7], df2.columns[:-1])
    ax.set_xlabel('Ranking')
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, y: r'$10^%d$' % x))
    plt.tight_layout()
    plt.savefig('rank-position-mcore-centrality-violin.pdf')


def rank_position_mcore_centrality_errbar(
        fn1='retweet.1108.claim.kcore.raw.csv',
        fn2='centralities.ranked.raw_id.csv'):
    """The position of main cores located at different centralities."""
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df2['rank_id'] = df2.index.values + 10
    df1 = df1.loc[df1.kcore == df1.kcore.max()]
    means = []
    errs = []
    for c in df2.columns[:-1]:
        rank_id = pd.merge(
            df1, df2, how='inner', left_on='raw_id', right_on=c).rank_id
        means.append(rank_id.mean())
        errs.append(rank_id.std() / np.sqrt(len(df1)))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        means,
        range(len(means)),
        xerr=errs,
        fmt='o',
        capsize=2,
        ecolor='red',
        color=C1)
    ax.set_xlabel('Ranking')
    plt.yticks([0, 1, 2, 3, 4, 5, 6], df2.columns[:-1])
    # ax.set_xscale('symlog')
    #ax.xaxis.set_major_formatter(
    #    mpl.ticker.FuncFormatter(lambda x, y: r'$10^%d$' % x))
    plt.tight_layout()
    plt.savefig('rank-position-mcore-centrality-errorbar.pdf')


def mcore_centrality_overlapping(fn1='retweet.1108.claim.kcore.raw.csv',
                                 fn2='centralities.ranked.raw_id.csv'):
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df2 = df2[['in_degree', 'out_degree']]
    df1 = df1.loc[df1.kcore == df1.kcore.max()]
    s1 = set(df1.raw_id.values)
    x = np.array(range(50, 2001, 50))
    y = []
    for top in x:
        s2 = set(df2.iloc[:top].values.flatten())
        y.append(len(s1 & s2) / len(s1))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y)
    ax.set_xlabel('Top Ranking')
    ax.set_ylabel('Inclusion')
    plt.tight_layout()


def mcore_growing(fn1='kcore.growing.csv',
                  fn2='kcore.growing.daily-rewiring.configuration.csv'):
    df1 = pd.read_csv(fn1, parse_dates=['timeline'])
    df2 = pd.read_csv(fn2, parse_dates=['timeline'])
    df1 = df1.set_index('timeline')
    df2 = df2.set_index('timeline')
    df1 = df1[['mcore_k', 'mcore_s']]
    df1.columns = ['K', 'S']
    df2 = df2[['mcore_k', 'mcore_s']]
    df2.columns = ['K, rewired', 'S, rewired']
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df = df.rolling('7D').mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df1.rolling('7D').mean().plot(
        ax=ax,
        secondary_y=['K'],
        color='r',
        marker='s',
        markersize=4,
        markevery=14,
        alpha=0.8)
    df2.rolling('7D').mean().plot(
        ax=ax,
        secondary_y=['K, rewired'],
        color='b',
        marker='o',
        markersize=4,
        markevery=14,
        alpha=0.8)


def mcore_growing_inset(fn1='kcore.growing.csv',
                        fn2='kcore.growing.daily-rewiring.configuration.csv'):
    """The main core size and K when growing of network."""
    df1 = pd.read_csv(fn1, parse_dates=['timeline'])
    df2 = pd.read_csv(fn2, parse_dates=['timeline'])
    df1 = df1.set_index('timeline')
    df2 = df2.set_index('timeline')
    df1 = df1[['mcore_k', 'mcore_s']]
    df1.columns = ['K, actual', 'S, actual']
    df2 = df2[['mcore_k', 'mcore_s']]
    df2.columns = ['K, shuffled', 'S, shuffled']
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df = df.rolling('7D').mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df[['K, actual', 'K, shuffled']].plot(ax=ax, color=[C1, C2])
    ax.set_xlabel('')
    ax.set_ylabel('K of Main Core')
    plt.tight_layout()
    ax2 = fig.add_axes([0.52, 0.24, 0.42, 0.31])
    df[['S, actual']].plot(ax=ax2, color=C1, fontsize=9, legend=False, lw=0.8)
    ax2.set_xlabel('')
    ax2.tick_params(
        axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax2.set_ylabel('Size', fontsize=9)
    ax2.set_yticks([0, 500, 1000])
    ax2.set_yticklabels(['0.0', '0.5', '1.0'])
    ax2.text(0.0, 1.01, '$x10^3$', fontsize=9, transform=ax2.transAxes)
    plt.savefig('mcore-growing-inset.pdf')


def bot_by_centrality(fn1='ubs_by_ometer.parsed.csv',
                      fn2='centralities.ranked.raw_id.csv',
                      top=1000):
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df2 = df2.iloc[:top]
    bmap = df1.set_index('user_raw_id').bot_score_en
    ms = []
    errs = []
    for c in df2.columns:
        vs = bmap.loc[df2[c].values]
        vs = vs.loc[vs.notnull()]
        ms.append(vs.mean())
        errs.append(vs.std() / np.sqrt(len(vs)))
        print((c, ms[-1], errs[-1]))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        range(len(ms)),
        ms,
        yerr=errs,
        ecolor='red',
        elinewidth=0.8,
        capsize=1.5,
        fmt='o-',
        markersize=1.2,
        color=C1,
        lw=0.8,)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(df2.columns, rotation=90)
    ax.set_ylabel('Bot Score')
    plt.tight_layout()
    plt.savefig('bot-by-centrality.pdf')


def bot_by_kcore(fn1='ubs_by_ometer.parsed.csv',
                 fn2='sampled.raw_id.by.kcore.csv'):
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df = pd.merge(
        df2, df1, how='left', left_on='raw_id', right_on='user_raw_id')
    df = df.loc[df.bot_score_en.notnull()]
    gp = df.groupby('k').bot_score_en
    m = gp.mean()
    yerr = gp.std() / np.sqrt(gp.size())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.plot(m.index.values, m.values,
    #         marker='o',
    #         markersize=4,
    #         alpha=0.5,
    #         )
    ax.errorbar(
        m.index.values,
        m.values,
        yerr=yerr,
        ecolor='red',
        elinewidth=0.8,
        capsize=1.5,
        fmt='o-',
        markersize=1.2,
        color=C1,
        lw=0.8,)
    ax.set_xlabel('K')
    ax.set_ylabel('Bot Score')
    plt.tight_layout()
    plt.savefig('bot-by-kcore.pdf')


def bot_by_kshell(fn1='ubs_by_ometer.parsed.csv',
                  fn2='sampled.raw_id.by.kshell.csv'):
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df = pd.merge(
        df2, df1, how='left', left_on='raw_id', right_on='user_raw_id')
    df = df.loc[df.bot_score_en.notnull()]
    gp = df.groupby('k').bot_score_en
    m = gp.mean()
    yerr = gp.std() / np.sqrt(gp.size())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.plot(m.index.values, m.values,
    #         marker='o',
    #         markersize=4,
    #         alpha=0.5,
    #         )
    ax.errorbar(
        m.index.values,
        m.values,
        yerr=yerr,
        ecolor='red',
        elinewidth=0.8,
        capsize=1.5,
        fmt='o-',
        markersize=1.2,
        color=C1,
        lw=0.8,)
    ax.set_xlabel('K')
    ax.set_ylabel('Bot Score')
    plt.tight_layout()
    plt.savefig('bot-by-kshell.pdf')

def changes_of_cores(fn='k_core_evolution.csv',
                     start='2016-09-01',
                     end='2017-10-01'):
    """The changes of mcores by intersection daily."""
    df = pd.read_csv(fn, parse_dates=['timeline'])
    df['mcore_idx'] = df.mcore_idx.apply(eval).apply(set)
    df = df.set_index('timeline')
    df.index = df.index.values.astype('<M8[D]')
    unchanged_core_num = []
    for dt in pd.date_range(start=start, end=end, freq='D'):
        if dt in df.index:
            s1 = set(df.loc[dt, 'mcore_idx'])
            for ts, s2 in df.mcore_idx.loc[(dt + pd.Timedelta('1 day')):].iteritems():
                s1 &= s2
            unchanged_core_num.append((dt, len(s1)))
    rdf = pd.DataFrame(unchanged_core_num, columns=['timeline', 'n'])
    rdf = rdf.set_index('timeline')
    rdf.plot()

