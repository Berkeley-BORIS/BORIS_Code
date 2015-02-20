from __future__ import absolute_import, print_function, division

from os.path import abspath, expanduser, exists
from os import makedirs

import click

from ..eyeparse import *
from ..utils import *
from ..framesync import *
from ..subject import *
from ..config import *

@click.group()
def main():
    #click.echo("Tesing the main function!")
    pass


@main.command()
@click.argument('subject_id')
@click.argument('task_id')
def parse(subject_id, task_id):
    """
    Parse an ascii eyelink file into a file with gaze info
    """

    subject = BORISSubject(subject_id)

    eye_data_parser = EyeDataParser(subject.raw_ascii_fpath(task_id))
    print("Parsing {subject_id} {task_id} using file {fpath}...".format(
        subject_id=subject_id, task_id=task_id, fpath=subject.raw_ascii_fpath(task_id)))
    eye_data_parser.parse_data()
    print("Done!\n")

    print("Creating DataFrames...")
    eye_dfs = EyeDataFrameCreator(eye_data_parser)
    print("Done!\n")

    print("Calculating target locations and fixations...")
    calc_target_locations(eye_dfs.radial_target_df, subject.ipd)
    print("Done!\n")

    if subject.needs_framesync(task_id):
        print("Subject {0} lost button presses during {1} task. "
              "Adding the missing frames...".format(subject.subject_id, task_id))
        eye_dfs.frame_df = fix_missing_frames(eye_dfs.frame_df, subject.framesync_fpath(task_id))
        print("Frames added.\n")

    print("Syncing radial target frames...")
    sync_frames(eye_dfs.radial_target_df, eye_dfs.frame_df)
    print("Syncing task frames...")
    sync_frames(eye_dfs.task_df, eye_dfs.frame_df)
    print("Done!\n")

    print("Saving data frames to {}".format(subject.gaze_data_fpath(task_id)))
    if not exists(subject.processed_gaze_dpath):
        makedirs(subject.processed_gaze_dpath)

    with pd.HDFStore(subject.gaze_data_fpath(task_id), 'w') as store:
        store['task'] = eye_dfs.task_df
        store['rt'] = eye_dfs.radial_target_df
        store['frames'] = eye_dfs.frame_df

    print("Parsing complete.\n")


@main.command()
@click.option('--root')
@click.option('--set', nargs=2)
def config(root, set):

    if root:
        rc['root_data_dpath']=abspath(expanduser(root))

    if set:
        rc[set[0]] = abspath(expanduser(set[1]))

    if root or set:
        write_boris_rc(rc, rc_path)
    else:
        print(rc)

# @main.command()
# def info():

#     template = /
# """
