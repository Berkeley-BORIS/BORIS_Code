"""
Classes to parse NDS files and create pandas DataFrames to hold eyetracking information.
"""

import numpy as np
import pandas as pd

class EyeDataParser(object):

    def __init__(self, data_fpath):

        self.data_fpath = data_fpath

        self.subject_id = 'XYZ'
        self.exp_id = 'NAT'

        self._task_data = {}
        self._task_rep = 0  # don't have this in the data files, so we keep track manually
        self._radial_target_data = {}
        self._frames = []
        self._frame_num = -1
        self._trash_data = []
        self._current_eye_flags = {'L':'NONE', 'R':'NONE'}
        self._current_data = self._trash_data

        self._RT_dist = None
        self._RT_rep = None
        self._RT_eccentricity = None
        self._RT_direction = None

    @property
    def task_data(self):
        return self._task_data

    @property
    def radial_target_data(self):
        return self._radial_target_data

    @property
    def frames(self):
        return self._frames

    def parse_data(self):

        with open(self.data_fpath) as f:
            for line in f:
                splitted_line = line.split()
                if not splitted_line:
                    continue
                line_type = splitted_line[0]

                # if line_type can convert to int, it's a DATA line
                try:
                    int(line_type)
                    line_type = 'DATA'
                except ValueError:
                    pass

                # execute the appropriate function
                # we might not know how to parse the line so we should create one if needed
                parse_func_name = '_parse_' + line_type
                try:
                    getattr(self, parse_func_name)(splitted_line)
                except AttributeError:
                    print "Passing unhandled line type", line_type
                    setattr(self, parse_func_name, self._parse_unknown_line)
                    getattr(self, parse_func_name)(splitted_line)

    def _parse_DATA(self, data):

        if data[-1] == '.....':
            data_quality = 'GOOD'
        else:
            data_quality = 'BAD'

        time = int(data[0])
        eye_data = [np.nan if i=='.' else float(i) for i in data[1:-1]]

        full_data = [time] + eye_data + [self._current_eye_flags['L'], self._current_eye_flags['R']] + [data_quality]
        RT_data = [self._RT_dist, self._RT_direction, self._RT_eccentricity, self._RT_rep]
        if not any(i is None for i in RT_data):  # if all RT data is valid
            full_data = full_data + RT_data  # extend the list with RT data
        self._current_data.append(full_data)

    def _parse_MSG(self, msg):

        msg_type = msg[2]
        if msg_type == 'RADIAL_TARGET_STARTED':
            self._RT_rep = int(msg[4])
            self._current_data = self._trash_data

        elif msg_type == 'RADIAL_TARGET':
            self._RT_dist = int(msg[3])
            self._RT_eccentricity = int(msg[4])
            self._RT_direction = int(msg[5])

            if int(msg[6]):
                rt_data = self._radial_target_data
                if not self._RT_rep in rt_data.keys():
                    rt_data[self._RT_rep] = []
                self._current_data = rt_data[self._RT_rep]
            else:
                self._current_data = self._trash_data

        elif msg_type == 'TASK_STARTED':
            self._task_rep += 1
            self._RT_eccentricity = None
            self._RT_direction = None
            self._RT_dist = None
            self._RT_rep = None

            task_data = self._task_data
            if not self._task_rep in task_data.keys():
                task_data[self._task_rep] = []
            self._current_data = task_data[self._task_rep]

        elif msg_type == 'TASK_STOPPED':
            self._current_data = self._trash_data

        elif msg_type == 'RADIAL_TARGET_STOPPED':
            self._current_data = self._trash_data
            self._RT_eccentricity = None
            self._RT_direction = None
            self._RT_dist = None
            self._RT_rep = None

    def _parse_SSACC(self, sacc):

        self._current_eye_flags[sacc[1]] = 'SACC'

    def _parse_ESACC(self, sacc):

        self._current_eye_flags[sacc[1]] = 'NONE'

    def _parse_SFIX(self, fix):

        self._current_eye_flags[fix[1]] = 'FIX'

    def _parse_EFIX(self, fix):

        self._current_eye_flags[fix[1]] = 'NONE'

    def _parse_SBLINK(self, blink):

        self._current_eye_flags[blink[1]] = 'BLINK'

    def _parse_EBLINK(self, blink):

        self._current_eye_flags[blink[1]] = 'NONE'

    def _parse_BUTTON(self, button):

        if int(button[3]):
            self._frame_num += 1
            time = int(button[1])
            self._frames.append([time, self._frame_num])

    def _parse_unknown_line(self, unk):

        pass

class EyeDataFrameCreator(object):

    def __init__(self, parser):

        self.task_df = self.create_task_df(parser.task_data)
        self.radial_target_df = self.create_rt_df(parser.radial_target_data)
        self.frame_df = self.create_frame_df(parser.frames)

    def create_task_df(self, task_data):

        col_arrays = [np.array(['left', 'left', 'left', 'right', 'right', 'right', 'left', 'right', 'both']),
                      np.array(['href x', 'href y', 'pupil area', 'href x', 'href y', 'pupil area', 'flag', 'flag', 'quality'])]

        task_df = self._generate_dataframe(task_data, col_arrays)

        return task_df

    def create_rt_df(self, rt_data):

        col_arrays = [np.array(['left', 'left', 'left', 'right', 'right', 'right', 'left', 'right', 'both', 'target', 'target', 'target', 'target']),
                      np.array(['href x', 'href y', 'pupil area', 'href x', 'href y', 'pupil area', 'flag', 'flag', 'quality', 'distance', 'direction', 'eccentricity', 'rep'])]

        rt_df = self._generate_dataframe(rt_data, col_arrays)

        return rt_df

    def _generate_dataframe(self, data_dict, columns):

        dfs = {}

        for rep, data in data_dict.iteritems():
            npdata = np.array(data)
            times = np.array(npdata[:,0], dtype=int)
            dfs[rep] = pd.DataFrame(data)

        full_df = pd.concat(dfs)
        full_df.reset_index(0, inplace=True)
        full_df.rename(columns={0:'time', 'level_0':'rep'}, inplace=True)
        full_df = full_df.set_index(['rep', 'time'])
        full_df.columns = columns
        full_df.sortlevel(axis=1)
        full_df.sortlevel(axis=0)

        return full_df

    def create_frame_df(self, frames):

        df = pd.DataFrame(frames, columns=('time', 'press num'))

        return df

if __name__ == '__main__':
    import os

    data_fpath = os.path.join(os.path.expanduser('~'), 'Desktop', 'DATA_FOR_NEW_IMAC', 'eyes',
        'data_processing', 'data', 'bwsnat5', 'bwsnat5.asc')

    eye_data_parser = EyeDataParser(data_fpath)
    print "Parsing..."
    eye_data_parser.parse_data()
    print "Done!\n"

    print "Creating DataFrames..."
    eye_dfs = EyeDataFrameCreator(eye_data_parser)
