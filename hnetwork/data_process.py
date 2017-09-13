import pandas as pd
import networkx as nx
import logging

logger = logging.getLogger()


def to_gml(efn, cfn, gml_fn):
    """Generate gml file that represent community detection results.

    Parameters
    ----------
    efn, edge list csv file
    cfn, community file, also should including vertex properties
    gml_fn, output gml file name
    """
    edf = pd.read_csv(efn)
    edf['weight'] = edf.weight.astype('int')
    edf.rename(columns=dict(from_id='fromId', to_id='toId'), inplace=True)
    cdf = pd.read_csv(cfn)
    cdf.set_index('idx', inplace=True)
    g = nx.from_pandas_dataframe(
        edf,
        source='fromId',
        target='toId',
        edge_attr=True,
        create_using=nx.DiGraph())
    nx.set_node_attributes(g, values=cdf.label.to_dict(), name='communityLabel')
    nx.set_node_attributes(g, values=cdf.uid.to_dict(), name='screenName')
    g = nx.convert_node_labels_to_integers(g)
    nx.write_gml(g, gml_fn)
