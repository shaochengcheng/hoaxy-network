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
    df['cweight'] = (df.weight * df.site_type).apply(round)
    df['fweight'] = df.weight - df.cweight
    fn_df = df.loc[df.cweight > 0].copy()
    fn_df = fn_df.drop(
        ['weight', 'fweight', 'site_type'], axis=1).rename(
            columns=dict(cweight='weight'))
    ff_df = df.loc[df.fweight > 0].copy()
    ff_df = ff_df.drop(
        ['weight', 'cweight', 'site_type'], axis=1).rename(
            columns=dict(fweight='weight'))
    logger.info('Saving fake news network to %r ...', fn_fn)
    fn_df.to_csv(fn_fn)
    logger.info('Saving fact checking network to %r ...', ff_fn)
    ff_df.to_csv(ff_fn)
