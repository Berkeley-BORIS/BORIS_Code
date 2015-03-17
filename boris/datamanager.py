from __future__ import absolute_import, print_function, division, unicode_literals

from os.path import join

from . import config

def get_raw_gaze_dpath(subject_id):
    return join(config.raw_gaze_dpath, subject_id)


def get_raw_ascii_fpath(subject_id, session_id):
    """Returns the file path to the raw ascii data for the task specified
    by task_id."""

    asc_fname = "{subject_id}_{task_id}.asc".format(subject_id=subject_id,
                                                    task_id=session_id)

    return join(get_raw_gaze_dpath(subject_id), asc_fname)


class DataManager(object):

    def __init__(self, root=None, **kwargs):

        if root is None:
            self.root = config.root_data_dpath
        else:
            self.root = root

        self._gaze_dname = kwargs.get('gaze_dname', config.gaze_dname)

    @property
    def gaze_dname(self):
        return self._gaze_dname

    def get_gaze_data_dpath(self, subject_id):

        return join(self.root, self.gaze_dname)

    def get_gaze_data_fpath(self, subject_id, xid):

        fname = "{subject_id}_{xid}.h5".format(subject_id=subject_id,
                                               xid=xid)

        return join(self.root, self.gaze_dname, fname)


class RawDataManager(DataManager):

    def __init__(self):

        root = join(config.root_data_dpath, config.raw_dname)
        super(RawDataManager, self).__init__(root)


class ProcessedDataManager(DataManager):

    def __init__(self):

        root = join(config.root_data_dpath, config.processed_dname)
        super(ProcessedDataManager, self).__init__(root)