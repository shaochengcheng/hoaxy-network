import logging

import graph_tool.all as gt
import networkx as nx
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


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

def v_betweenness(g, fn):
    vp, ep = gt.betweenness(g)
    vb = {g.vertex(i):vp[g.vertex(i) for i in range(g.num_vertices())]}
    s = pd.Series(vb)
    s.name = 'v_betweenness'
    s.index.name = 'raw_id'
    s = s.sort_values(ascending=False)
    s.to_csv(fn)


def v_percolate(g, ofn, vertices):
    vertices = list(vertices)
    sizes, comp = gt.vertex_percolation(g, vertices)
    np.random.shuffle(vertices)
    sizes2, comp = gt.vertex_percolation(g, vertices)
    plt.figure()
    plot(sizes, label='Targeting')
    plot(sizes2, lable='Random')
    plt.savefig(ofn)



