"""This module provides ways to pre-process the data."""

import numpy as np
from graph_tool import GraphView, load_graph
from graph_tool.draw import (graph_draw, sfdp_layout,
                             fruchterman_reingold_layout, arf_layout,
                             random_layout, prop_to_size)
from graph_tool.util import find_vertex
import logging

logger = logging.getLogger()

MIN_V_SIZE = 1
MAX_V_SIZE = 15
V_SIZE_LOG = False
V_SIZE_POWER = 0.5

MIN_E_WIDTH = 0.2
MAX_E_WIDTH = 5
E_WIDTH_LOG = False
E_WIDTH_POWER = 0.5


def draw_community(gml_fn,
                   output,
                   layout_name=None,
                   layout_kwargs=dict(),
                   **draw_kwargs):
    g = load_graph(gml_fn)

    # Sampel of graph g
    # g = GraphView(g, vfilt=lambda v: g.vertex_index[v]%2==0)
    g.vp['wdeg'] = g.degree_property_map('total', weight=g.ep['weight'])
    # g = GraphView(g, vfilt=lambda v: g.vp['wdeg'][v]>0)

    # label for hub account only in each community
    g.vp['clabel'] = g.new_vertex_property("string", val="")
    for c in np.nditer(np.unique(g.vp['community'].a)):
        cg = GraphView(g, vfilt=(g.vp['community'].a == c))
        v_hub = find_vertex(cg, cg.vp['wdeg'], cg.vp['wdeg'].fa.max())[0]
        cg.vp['clabel'][v_hub] = cg.vp['screenname'][v_hub]

    v_size = prop_to_size(
        g.vp['wdeg'],
        mi=MIN_V_SIZE,
        ma=MAX_V_SIZE,
        log=V_SIZE_LOG,
        power=V_SIZE_POWER)
    e_width = prop_to_size(
        g.ep['weight'],
        mi=MIN_E_WIDTH,
        ma=MAX_E_WIDTH,
        log=E_WIDTH_LOG,
        power=E_WIDTH_POWER)
    if layout_name is not None:
        try:
            pos = globals()[layout_name](g, **layout_kwargs)
        except KeyError as e:
            logger.critical('No such layout function found!')
            raise
    graph_draw(
        g,
        pos,
        output=output,
        vprops=dict(
            fill_color=g.vp['community'],
            # color='grey',
            size=v_size,
            pen_width=0.01,
            text=g.vp['clabel'],
            text_position='centered',
            font_size=8,),
        eprops=dict(
            pen_width=e_width,
            end_marker="arrow",),
        **draw_kwargs)
