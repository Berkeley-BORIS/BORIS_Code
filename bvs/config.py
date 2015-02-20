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

DEFAULT_RAW_DNAME = 'raw'
DEFAULT_GAZE_DNAME = 'gaze'
DEFAULT_REGISTRATION_DNAME = 'cam2eye_registration'
DEFAULT_SCENE_DNAME = 'scene'
DEFAULT_STEREOCALIB_DNAME = 'stereocalibration'

root_data_dpath = rc.get('root_data_dpath', None)

if root_data_dpath is None:
    print("WARNING: BORIS has not been configured. Please run\n" +
        "boris config --root=path/to/data/root\n" +
        "to tell BORIS where the data is located.")

raw_dname = rc.get('raw_dname', DEFAULT_RAW_DNAME)

gaze_dname = rc.get('gaze_dname', DEFAULT_GAZE_DNAME)
registration_dname = rc.get('registration_dname', DEFAULT_REGISTRATION_DNAME)
scene_dname = rc.get('scene_dname', DEFAULT_SCENE_DNAME)
stereocalibration_dname = rc.get('stereocalibration_dname', DEFAULT_STEREOCALIB_DNAME)



