from __future__ import absolute_import, print_function, division

import click

from bvs.eyelinkfileparsing import *

@click.command()
def main():
    click.echo("Tesing the main function!")


def parse_eyelink(fpath):
    """
    Parse an ascii eyelink file into a file 