from __future__ import absolute_import, print_function, division

import click

from bvs.eyeparse import *
from bvs.utils import *

@click.group()
def main():
    #click.echo("Tesing the main function!")
    pass


@main.command()
@click.argument('fpath')#, help='Path to the EyeLink ascii file.')
def parse(fpath):
    """
    Parse an ascii eyelink file into a file with gaze info
    """

    import os
    from glob import glob

    dpath, fname = os.path.split(fpath)
    print(dpath)
    ipd_fpath = glob(os.path.join(dpath, '*_ipd.txt'))[0]

    with open(ipd_fpath, 'r') as f:
        ipd = float(f.read())

    eye_data_parser = EyeDataParser(fpath)
    print("Parsing...")
    eye_data_parser.parse_data()
    print("Done!\n")

    print("Creating DataFrames...")
    eye_dfs = EyeDataFrameCreator(eye_data_parser)
    print("Done!\n")

    print("Calculating target locations and fixations...")
    calc_target_locations(eye_dfs.radial_target_df, ipd)

    print("Done!\n")

    print("Syncing radial target frames...")
    sync_frames(eye_dfs.radial_target_df, eye_dfs.frame_df)
    print("Syncing task frames...")
    sync_frames(eye_dfs.task_df, eye_dfs.frame_df)
    print("Done!\n")

    # TODO Decide where to put the parsed data. Putting it in a test file for now.
    # Probably want to make a list of current subjects and tasks
    out_fpath = './testing/parser_output'
    print("Saving data frames to {}".format(out_fpath))
    if not os.path.exists(out_fpath):
	    os.makedirs(out_fpath)
    eye_dfs.task_df.to_hdf(os.path.join(out_fpath, 'task_output.h5'), 'task')
    eye_dfs.radial_target_df.to_hdf(os.path.join(out_fpath, 'rt_output.h5'), 'rt')
    eye_dfs.frame_df.to_hdf(os.path.join(out_fpath, 'frame_output.h5'), 'frames', index=False)
    print("Parsing complete.\n")