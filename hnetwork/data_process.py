import logging
from os.path import join, splitext, basename

import pandas as pd
import networkx as nx
import numpy as np

from . import BASE_DIR

logger = logging.getLogger()

DATA_DIR = join(BASE_DIR, 'data')
PLOTS_DIR = join(BASE_DIR, 'plots')


def get_data_file(fn):
    return join(DATA_DIR, fn)


def get_out_file(fn):
    return join(PLOTS_DIR, fn)


def get_absprefix(abspath):
    root, ext = splitext(abspath)
    return root


def nplog(a, base):
    return np.log(a)/np.log(base)


def ccdf(s):
    """
    Parameters:
        `s`, series, the values of s should be variable to be handled
    Return:
        a new series `s`, index of s will be X axis (number), value of s
        will be Y axis (probability)
    """
    s = s.copy()
    s = s.sort_values(ascending=True, inplace=False)
    s.reset_index(drop=True, inplace=True)
    n = len(s)
    s.drop_duplicates(keep='first', inplace=True)
    X = s.values
    Y = [n - i for i in s.index]

    return pd.Series(data=Y, index=X) / n


def decompose_network(fn):
    """Return the decomposed network by site_type."""
    fn = get_data_file(fn)
    logger.info('Raw network file is: %r', fn)
    logger.info('Decomposing ...')
    fbn, fext = splitext(basename(fn))
    fn_fn = get_data_file(fbn + '.fn.csv')
    ff_fn = get_data_file(fbn + '.fc.csv')
    df = pd.read_csv(fn)
    if 'from_indexed_id' in df.columns:
        df = df.drop('from_indexed_id', axis=1)
    if 'to_indexed_id' in df.columns:
        df = df.drop('to_indexed_id', axis=1)
    fn_df = df.loc[df.cweight > 0].copy()
    fn_df = fn_df.drop('fweight', axis=1).rename(
            columns=dict(cweight='weight'))
    ff_df = df.loc[df.fweight > 0].copy()
    ff_df = ff_df.drop('cweight', axis=1).rename(
            columns=dict(fweight='weight'))
    logger.info('Saving fake news network to %r ...', fn_fn)
    fn_df.to_csv(fn_fn, index=False)
    logger.info('Saving fact checking network to %r ...', ff_fn)
    ff_df.to_csv(ff_fn, index=False)


def index_edge_list(ifn, ofn, ikwargs=dict(), okwargs=dict(index=False)):
    df = pd.read_csv(ifn, **ikwargs)
    s = pd.concat((df[df.columns[0]], df[df.columns[1]]))
    s = s.drop_duplicates()
    s = s.sort_values().reset_index(drop=True)
    s = {v:k for k, v in s.to_dict().items()}
    df['from_idx'] = df[df.columns[0]].apply(s.get)
    df['to_idx'] = df[df.columns[1]].apply(s.get)
    df = df.sort_values(['from_idx', 'to_idx'])
    df[['from_idx', 'to_idx']].to_csv(ofn, **okwargs)

