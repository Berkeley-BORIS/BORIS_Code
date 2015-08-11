from __future__ import absolute_import, print_function, division

from os.path import abspath, expanduser, exists
from os import makedirs,listdir
import subprocess
from fnmatch import fnmatch

import click

from ..eyeparse import *
from ..stereocalibration import *
from ..utils import *
from ..framesync import *
from ..subject import *
from ..config import *
from ..datamanager import *
from ..aligneyes import *

@click.group()
def main():
    #click.echo("Tesing the main function!")
    pass


@main.command()
@click.argument('subject_id')
@click.argument('session_id')
def parse(subject_id, session_id):
    """
    Parse an ascii eyelink file into a file with gaze info
    """

    subject = BORISSubject(subject_id)

    eye_data_parser = EyeDataParser(subject.raw_ascii_fpath(session_id))
    print("Parsing {subject_id} {session_id} using file {fpath}...".format(
        subject_id=subject_id, session_id=session_id, fpath=subject.raw_ascii_fpath(session_id)))
    eye_data_parser.parse_data()
    print("Done!\n")

    print("Creating DataFrames...")
    eye_dfs = EyeDataFrameCreator(eye_data_parser)
    print("Done!\n")

    print("Calculating target locations and fixations...")
    calc_target_locations(eye_dfs.radial_target_df, subject.ipd)
    print("Done!\n")

    if subject.needs_framesync(session_id):
        print("Subject {0} lost button presses during {1} task. "
              "Adding the missing frames...".format(subject.subject_id, session_id))
        eye_dfs.frame_df = fix_missing_frames(eye_dfs.frame_df, subject.framesync_fpath(session_id))
        print("Frames added.\n")

    print("Syncing radial target frames...")
    sync_frames(eye_dfs.radial_target_df, eye_dfs.frame_df)
    print("Syncing task frames...")
    sync_frames(eye_dfs.task_df, eye_dfs.frame_df)
    print("Done!\n")

    print("Calculating pupil sizes...")
    calc_metric_pupil_size(subject.pupil_size, eye_dfs.task_df)
    print("Done!\n")

    print("Saving data frames to {}".format(subject.gaze_data_fpath(session_id)))
    if not exists(subject.processed_gaze_dpath):
        makedirs(subject.processed_gaze_dpath)

    with pd.HDFStore(subject.gaze_data_fpath(session_id), 'w') as store:
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
# """o

@main.command()
@click.argument('subject_id')
@click.argument('session_id')
@click.argument('task_id')
def original_eye_analysis(subject_id, session_id, task_id):

    # instantiating BORIS subject class
    subject = BORISSubject(subject_id)
    # manage paths to session level (non project-specific data)
    session_dm = ProcessedDataManager()
    # manage paths to project-specific data home
    nds_dm = DataManager(root=nds_root_dpath, scene_data_root=nds_task_root)

    with pd.HDFStore(session_dm.get_gaze_data_fpath(subject_id, session_id),
                     mode='r') as store:
        session_gaze_data = store['task']
        rt_data = store['rt']

    print("Extracting task...")
    task_gaze_data = extract_task(session_gaze_data,
                        nds_dm.get_scene_data_dpath(subject_id, task_id))

    print("Calculating task gaze positions...")
    convert_href_to_bref(task_gaze_data, rt_data.copy())
    align_eyes(task_gaze_data, subject.ipd)
    calc_fixation_pts(task_gaze_data, subject.ipd)  # TODO This is failing to detect diverging eyes?
    calc_version(task_gaze_data)
    calc_vergence(task_gaze_data, subject.ipd)

    print("Calculating rt gaze positions...")
    convert_href_to_bref(rt_data, rt_data.copy())
    align_eyes(rt_data, subject.ipd)
    calc_fixation_pts(rt_data, subject.ipd)
    calc_version(rt_data)
    calc_vergence(rt_data, subject.ipd)

    print("Saving dataframes to {}".format(nds_dm.get_gaze_data_fpath(subject_id, task_id)))
    if not exists(nds_dm.get_gaze_data_dpath(subject_id)):
        makedirs(nds_dm.get_gaze_data_dpath(subject_id))

    with pd.HDFStore(nds_dm.get_gaze_data_fpath(subject_id, task_id), 'w') as store:
        store['task'] = task_gaze_data
        store['rt'] = rt_data


@main.command()
def parse_all():
    """Parse all subjects and all sessions"""

    for subject_id in ['kre', 'sah', 'tki']:
        for session_id in ['cafe', 'inside', 'nearwork', 'outside1', 'outside2']:
            cmd = "boris parse {subject_id} {session_id}".format(subject_id=subject_id,
                                                                 session_id=session_id)
            subprocess.call(cmd, shell=True)


@main.command()
@click.argument('subject_id')
@click.argument('session_id')
def stereocalibrate(subject_id, session_id):

    """
    Estimate intrinsics and extrinsics of stereo camera rig
    """
    
    subject = BORISSubject(subject_id)

    # grab calibration frames directory
    for dir in listdir(subject.stereocalibration_session_dpath(session_id)):
        if fnmatch(dir,'calibration_frames*'):
            check_img_folder = join(subject.stereocalibration_session_dpath(session_id),dir)
            break

    print("Stereo-Calibrating {subject_id} {session_id} using file {fpath}...".format(
        subject_id=subject_id, session_id=session_id, fpath=check_img_folder))
    

    stereo_calibrator = StereoCalibrator(check_img_folder)
    stereo_calibrator.calibrate()
    print("Done!\n")

