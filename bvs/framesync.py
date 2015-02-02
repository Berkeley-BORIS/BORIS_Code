from __future__ import absolute_import, print_function, division

import yaml
import numpy as np
import pandas as pd

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


def fix_missing_frames(frame_df, framesync_fpath):
    """
    Load in the framesync file to add frames with missing button presses back
    into the frames dataframe.
    """

    with open(framesync_fpath, 'r') as f:

        sync_data = yaml.load(f)

        for d in sync_data:
            print("Fixing missing buttons in {}".format(d['name']))
            print("Using {} as reference.".format(d['reference_name']))

            new_startframe = d['radial_targets_starttime'] + d['capture_start_offset']
            new_captures = frame_df[(frame_df['time'] >= d['reference_starttime']) &
                                    (frame_df['time'] <= d['reference_endtime'])].copy()
            new_captures -= new_captures.iloc[0]
            new_captures['press num'] += 1 + frame_df['press num'].iloc[-1]
            new_captures['time'] += new_startframe

            extra_buttons = [new_captures['press num'].iloc[-1] + i
                            for i in xrange(1,1+d['num_additional_frames'])]
            extra_frame_times = [new_captures['time'].iloc[-1] + i*33
                                for i in xrange(1,1+d['num_additional_frames'])]

            extra_frames = pd.DataFrame({'time':extra_frame_times, 'press num': extra_buttons})
            extra_frames = extra_frames[extra_frames['time'] < d['radial_targets_endtime']]
            frame_df = pd.concat([frame_df, new_captures, extra_frames])

    return frame_df




