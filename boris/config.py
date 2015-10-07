"""
Configuration loader for BORIS.
"""

from __future__ import absolute_import, print_function, division, unicode_literals
from os.path import join, expanduser, isfile
import yaml

rc_path = join(expanduser("~"), ".borisrc")

def load_boris_rc(fpath):

    if not fpath or not isfile(fpath):
        return {}

    return yaml.load(open(fpath, 'r'))

def write_boris_rc(rc, fpath):

    f = open(fpath, 'w')  # currently this completely overwrites the rc file
    f.write(yaml.safe_dump(rc, default_flow_style=False))
    f.close()

rc = load_boris_rc(rc_path)

_DEFAULT_RAW_DNAME = 'raw'
_DEFAULT_PROCESSED_DNAME = 'processed'
_DEFAULT_GAZE_DNAME = 'gaze'
_DEFAULT_REGISTRATION_DNAME = 'cam2eye_registration'
_DEFAULT_SCENE_DNAME = 'scene'
_DEFAULT_STEREOCALIB_DNAME = 'stereocalibration'

root_data_dpath = rc.pop('root_data_dpath', None)

if root_data_dpath is None:
    print("WARNING: BORIS has not been configured. Please run\n" +
        "boris config --root=path/to/data/root\n" +
        "to tell BORIS where the data is located.")
else:
    raw_dname = rc.pop('raw_dname', _DEFAULT_RAW_DNAME)

    gaze_dname = rc.pop('gaze_dname', _DEFAULT_GAZE_DNAME)
    registration_dname = rc.pop('registration_dname', _DEFAULT_REGISTRATION_DNAME)
    scene_dname = rc.pop('scene_dname', _DEFAULT_SCENE_DNAME)
    stereocalibration_dname = rc.pop('stereocalibration_dname', _DEFAULT_STEREOCALIB_DNAME)
    cam2eye_registration_dname = rc.pop('scam2eye_registration_dname', _DEFAULT_REGISTRATION_DNAME)

    raw_gaze_dpath = rc.pop('raw_gaze_dpath', join(root_data_dpath, raw_dname, gaze_dname))
    registration_dpath = rc.pop('registration_dpath',
        join(root_data_dpath, raw_dname, registration_dname))
    raw_scene_dpath = rc.pop('raw_scene_dpath',
        join(root_data_dpath, raw_dname, scene_dname))
    stereocalibration_dpath = rc.pop('stereocalibration_dpath',
        join(root_data_dpath, raw_dname, stereocalibration_dname))
    cam2eye_registration_dpath = rc.pop('cam2eye_registration_dpath',
        join(root_data_dpath, raw_dname, cam2eye_registration_dname))

    processed_dname = rc.pop('processed_dname', _DEFAULT_PROCESSED_DNAME)

    processed_gaze_dpath = rc.pop('processed_dpath',
        join(root_data_dpath, processed_dname, gaze_dname))

    processed_stereocalibration_dpath = rc.pop('processed_dpath',
        join(root_data_dpath, processed_dname, stereocalibration_dname))

    processed_cam2eye_registration_dpath = rc.pop('processed_dpath',
        join(root_data_dpath, processed_dname, cam2eye_registration_dname))

    processed_scene_dpath = rc.pop('processed_scene_dpath',
        join(root_data_dpath, processed_dname, scene_dname))

    for key in rc.keys():
        locals()[key] = rc.pop(key)

