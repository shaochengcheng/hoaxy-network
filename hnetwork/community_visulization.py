"""This module provides ways to pre-process the data."""

import pandas as pd
import networkx as nx
from graph_tool import load_graph
from graph_tool.draw import *
import logging

logger = logging.getLogger()


def draw_community(gml_fn, output,
                   layout_name=None,
                   layout_kwargs=dict()
                   **draw_kwargs):
    g = load_graph(gml_fn)
    if layout_name is not None:
        try:
            pos = globals()[layout_name](**layout_kwargs)
        except KeyError as e:
            logger.critical('No such layout function found!')
            raise
    graph_draw(g, pos, output=output,
               vprops=dict(
                   fill_color=g.vp['community'],
                   size=1,
                   pen_width=0.1,
                   text="",
                   text_position=-1,
                   font_size=9,
               ),
               eprops=dict(
                   pen_width=1.0,
                   end_marker="arrow",
               )
               **draw_kwargs)







