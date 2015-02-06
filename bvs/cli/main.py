from __future__ import absolute_import, print_function, division

import click

from bvs.eyeparse import *
from bvs.utils import *
from ..framesync import *
from ..subject import *

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
    print("Parsing...")
    eye_data_parser.parse_data()
    print("Done!\n")

    print("Creating DataFrames...")
    eye_dfs = EyeDataFrameCreator(eye_data_parser)
    print("Done!\n")

    print("Calculating target locations and fixations...")
    calc_target_locations(eye_dfs.radial_target_df, subject.ipd)
    print("Done!\n")

    if subject.needs_framesync(task_id):
        print("Subject {0} lost button presses during {1} task. " +  # FIXME Not sure this will work
              "Adding the missing frames...".format(subject.subject_id, task_id))
        eye_dfs.frame_df = fix_missing_frames(eyedfs.frame_df, subject.framesync_fpath(task_id))
        print("Frames added.\n")

    print("Syncing radial target frames...")
    sync_frames(eye_dfs.radial_target_df, eye_dfs.frame_df)
    print("Syncing task frames...")
    sync_frames(eye_dfs.task_df, eye_dfs.frame_df)
    print("Done!\n")

    print("Saving data frames to {}".format(subject.parsed_dpath))
    eyedfs.task_df.to_hdf(subject.task_fpath(task_id), 'task')
    eyedfs.radial_target_df.to_hdf(subject.radial_target_fpath(task_id), 'rt')
    eyedfs.frame_df.to_hdf(subject.frame_fpath(task_id), 'frames', index=False)
    print("Parsing complete.\n")