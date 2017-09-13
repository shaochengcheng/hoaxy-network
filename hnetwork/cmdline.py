"""
Usage:
    hnetwork viscomm [--force] <in_gml_file> [<out_pdf_dir>]

hnetwork is a tool to do the network analysis of hoaxy data. This command line
will provide commands to visulize community detection results.
Subcommands:
    viscomm         community visualization.

The community visualization subcommand will generate several outputs (pdf) with
different layout. The output name is the concatnation of input and layout name.
And options are:
    --force         Run visualization even if the output file exists.

Examples:
    1. Community visualiztion.
        hnetwork viscomm
"""
import logging
import sys
from os import pardir, makedirs
from os.path import join, basename, abspath, splitext, isfile, split

from docopt import docopt
from schema import (And, Or, Schema, SchemaError, Use)

from . import VERSION
from .community_visualization import draw_community

logger = logging.getLogger()


def main(argv=None):
    """The main entry point of command line interface."""
    args_schema = Schema({
        object: object
        })
    args = docopt(__doc__, version=VERSION, argv=argv or sys.argv[1:])
    try:
        args = args_schema.validate(args)
    except SchemaError as e:
        raise SystemExit(e)
    # set logging level
    logging.basicConfig(stream=sys.stdout, level='DEBUG')
    if args['viscomm'] is True:
        layouts = ['sfdp_layout', 'fruchterman_reingold_layout', 'arf_layout',
                'random_layout']
        gml_fn = abspath(args['<in_gml_file>'])
        gml_dir, gml_basename = split(gml_fn)
        pdf_dir = args['<out_pdf_dir>']
        if pdf_dir is None:
            pdf_dir = gml_dir
        else:
            pdf_dir = abspath(pdf_dir)
            makedirs(pdf_dir, mode=0o700, exist_ok=True)
        base_file, extention = splitext(gml_basename)
        logging.info('Input GML file is: %r', gml_fn)
        logging.info('Output PDF directory is: %r', pdf_dir)
        for layout in layouts:
            logger.info('Using layout name: %r', layout)
            pdf_fn = join(pdf_dir, base_file + '.' + layout + '.pdf')
            logger.info('Output PDF file for this layout is: %r', pdf_fn)
            if args['--force'] is False:
                try:
                    with open(pdf_fn, 'rb') as f:
                        pass
                    logger.warning('Output PDF file %r exists, skip', pdf_fn)
                    continue
                except FileNotFoundError as e:
                    pass
            logger.info('Now visulizing with layout %r...', layout)
            try:
                draw_community(gml_fn, pdf_fn, layout_name=layout)
            except FileNotFoundError as e:
                raise
            except Exception as e:
                logger.error(e)
            logger.info('Visulization with layout %r done.', layout)
    else:
        raise SystemExit('Invalid subcommand, try `hoaxy-tool -h`')
