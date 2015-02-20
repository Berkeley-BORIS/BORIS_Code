"""
Methods that process the eye movement data, including epipolar reprojection,
calculating version, vergence, torsion, and the 3D fixation point.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .physical import calibration_dist, HREF_DIST

_DEG_PER_RAD = 180.0/np.pi
_RAD_PER_DEG = np.pi/180.0


def calc_target_locations(df, ipd):

    target_directions = df['target', 'direction']
    target_eccentricities = df['target', 'eccentricity']
    target_dists = df['target', 'distance']

    # target x and y positions in world coords (z position is just target dist)
    target_hypotenuses = target_dists * np.tan(target_eccentricities*_RAD_PER_DEG)
    x = target_hypotenuses * np.cos(target_eccentricities*_RAD_PER_DEG)
    y = target_hypotenuses * np.sin(target_eccentricities*_RAD_PER_DEG)

    df['target', 'x'] = x
    df['target', 'y'] = y

    target_expected_fixation_pts = np.c_[x, y, target_dists]

    vergences = calc_vergence(target_expected_fixation_pts, ipd)
    versions = calc_version(target_expected_fixation_pts)

    df['target', 'horizontal version'] = versions[0]
    df['target', 'vertical version'] = versions[1]


def calc_fixation_pts(task_df, rt_df, ipd):

    href_center = _find_href_center(rt_df)
    for df in [task_df, rt_df]:
        for eye in ['left', 'right']:
            _convert_href_to_ndsref(df, eye=eye, center=href_center)
        _ndsref_to_fixation(df, ipd)


def _find_href_center(rt_df):

    central_target = (rt_df['target', 'direction'] == 0) & \
                     (rt_df['target', 'eccentricity'] == 0) & \
                     (rt_df['target', 'distance'] == calibration_dist) & \
                     (rt_df['target', 'rep'] == 1)

    good_data = rt_df['both', 'quality'] == 'GOOD'

    central_data = rt_df[[('left', 'href x'), ('left', 'href y'),
                          ('right', 'href x'), ('right', 'href y')]][central_target & good_data]

    return central.mean()


def _convert_href_to_ndsref(df, eye, center):

    CM_PER_HREF_UNIT = calibration_dist / HREF_DIST
    # recenter href coordinates and convert to cm to get ndsref for each eye
    # in the world coordinate system
    df[eye, 'ndsref x'] = (df[eye, 'href x'] - center[eye, 'href x']) * CM_PER_HREF_UNIT
    df[eye, 'ndsref y'] = (df[eye, 'href y'] - center[eye, 'href y']) * CM_PER_HREF_UNIT

    # flip y-axis so up is positive
    df[eye, 'ndsref y'] *= -1

    df[eye, 'ndsref z'] = calibration_dist


def _ndsref_to_fixation(df, ipd):

    ndsref = df[[('left', 'ndsref x'), ('left', 'ndsref y'), ('left', 'ndsref z'),
                 ('right', 'ndsref x'), ('right', 'ndsref y'), ('right', 'ndsref z')]]

    eyeref_le = ndsref['left'] - np.array([-ipd/2.0, 0, 0])
    eyeref_re = ndsref['right'] - np.array([ipd/2.0, 0, 0])

    # TODO fit eyerefs vectors to plane and project them
    _fit_to_plane()

    assert 0


def sync_frames(data_df, frames):
    """
    Synchronize the camera frames to the eyetracker data. The algorithm proceeds as follows:
    *For each frame
    *Grab the timestamp and capture number of the frame
    *Find indices of time points within 30ms window (+/- 15ms) of frame capture
    *Grab GOOD fixation thats closest to shutter open
    *Assign frame to that gaze position

    For tkidrdp1 (tki_inside) and bwsure1 (kre_outside2) the last two radial targets lost their
    button presses, but the cameras kept running, so we have the images. We can sync these manually
    by back-calculating where the frames came from. The code to do this should fix up the "frames"
    dataframe before being passed to this function.
    """

    # Find earliest and latest times contained within the data df. No point in
    # looking at frames that are outside the data.
    earliest_time = data_df.index.levels[1].min()
    latest_time = data_df.index.levels[1].max()
    task_frames = frames[(frames['time']>earliest_time) & (frames['time']<latest_time)]

    for i, frame in task_frames.iterrows():

        # For each frame find the time differences between it and the
        # eyetracker data and give it the same index as the data.
        time_diffs = data_df.index.levels[1] - frame['time']
        time_diffs = time_diffs.to_series().reset_index(drop=True)
        time_diffs.index = data_df.index

        # Mask out those times that are more than 15 ms from the frame time.
        time_range_mask = np.abs(time_diffs) < 15
        data_of_interest = data_df.loc[time_range_mask]
        # We only want to assign frames to GOOD eyemovements.
        good_data = data_of_interest[data_of_interest['both', 'quality'] == 'GOOD']

        # If we have any eyemovements within 15ms of the frame
        if not data_of_interest.empty:
            # Add their time diffs to the data df so we can spot check.
            indices_of_interest = data_of_interest.index
            data_df.loc[indices_of_interest, ('both', 'frame_time_diff')] = time_diffs

        # If any eyemovements with 15ms of the frame are also GOOD
        indices_of_interest = good_data.index
        if not time_diffs[indices_of_interest].empty:

            # Find the eyemovement closest to the frame and assign the frame to
            # that eyemovement.
            smallest_time_diff_loc = np.abs(time_diffs.loc[indices_of_interest]).argmin()
            data_df.loc[smallest_time_diff_loc, ('both', 'frame_time')] = frame['time']
            data_df.loc[smallest_time_diff_loc, ('both', 'frame_count')] = frame['press num']


def calc_vergence(fixation_pt, ipd):

    fixation_pt = np.atleast_2d(fixation_pt)
    assert fixation_pt.shape[1] == 3
    eyeref_le = fixation_pt - [-ipd/2.0, 0, 0]
    eyeref_re = fixation_pt - [ipd/2.0, 0, 0]

    vergence = calc_angle(eyeref_le, eyeref_re)

    nan_inds = np.where(np.isnan(fixation_pt).any(axis=1))
    vergence[nan_inds] = np.nan

    return vergence


def calc_version(fixation_pt):

    fixation_pt = np.atleast_2d(fixation_pt)
    assert fixation_pt.shape[1] == 3
    z_dir = np.array([0, 0, 1.0])

    zy_component = fixation_pt.copy()
    zy_component[:,0] = 0

    vert_version = calc_angle(z_dir, zy_component) * get_angle_direction(z_dir, zy_component, axis='x')
    horz_version = calc_angle(zy_component, fixation_pt) * get_angle_direction(fixation_pt, zy_component, axis='y')

    # flag bad/missing data with nans
    nan_inds = np.where(np.isnan(fixation_pt).any(axis=1))
    vert_version[nan_inds] = np.nan
    horz_version[nan_inds] = np.nan

    return (horz_version, vert_version)


def calc_angle(v1, v2):
    """
    Calculate the angle between two sets of vectors v1 and v2 in degrees. Inputs
    must both be Nx3 unless one argument is 1x3 or (3,).
    """

    v1, v2 = np.atleast_2d(v1, v2)  # vectors will be (1,3) or (N,3)
    assert v1.shape[1] == v2.shape[1] == 3

    v1_norm = np.sqrt(np.sum(v1**2, axis=1))
    v2_norm = np.sqrt(np.sum(v2**2, axis=1))

    v1N = v1/np.expand_dims(v1_norm, axis=1)
    v2N = v2/np.expand_dims(v2_norm, axis=1)

    cosarg = np.sum(v1N*v2N, axis=1)
    cosarg = np.min(np.c_[np.ones((cosarg.shape[0], 1)), cosarg], axis=1)

    return np.arccos(cosarg) * _DEG_PER_RAD


def get_angle_direction(v1, v2, axis):
    """
    Return the direction of rotation to get v1 onto v2 about the provided axis.
    """

    axes_dict = {'x':0, 'y':1, 'z':2, 'X':0, 'Y':1, 'Z':2}
    if axis in axes_dict:
        axis = axes_dict[axis]

    v1, v2 = np.atleast_2d(v1, v2)  # vectors will be (1,3) or (N,3)

    vec_normal = np.cross(v1, v2)

    directions = np.where(vec_normal[:,axis] == 0, np.ones(vec_normal.shape[0]), np.sign(vec_normal[:,axis]))

    return directions


def get_R(theta, phi):
    """
    Convert horizontal and vertical versions into a rotation matrix. phi is the vertical rotation
    and is positive DOWNWARD, theta is the horizontal rotation and is positive LEFTWARD.
    """

    phi = phi * _RAD_PER_DEG #phi is the angle of rotation about the x axis (ie vertical version)
    theta = theta * _RAD_PER_DEG # theta is the angle about the y-axis (ie horizontal version)

    R_phi = np.matrix([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi), np.cos(phi)]])
    R_theta = np.matrix([[np.cos(theta), 0, -np.sin(theta)],
                         [0, 1, 0],
                         [np.sin(theta), 0, np.cos(theta)]])

    R = R_phi*R_theta

    return R.A