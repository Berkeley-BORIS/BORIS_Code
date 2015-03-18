from __future__ import absolute_import, print_function, division, unicode_literals

from os.path import join
from glob import glob

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
    """Class to manage paths to all the different data BORIS needs"""
    # NOTE This currently relies on the initialization procedure to set the
    # directory names and paths to appropriate values. All methods should
    # reference only member variables, not config directly.

    def __init__(self, root=None, **kwargs):

        if root is None:
            self._root = config.root_data_dpath
        else:
            self._root = root

        self._gaze_dname = kwargs.get('gaze_dname', config.gaze_dname)
        self._gaze_data_root = kwargs.get('gaze_data_root', self._root)
        self._scene_dname = kwargs.get('scene_dname', config.scene_dname)
        self._scene_data_root = kwargs.get('scene_data_root', self._root)

    @property
    def gaze_dname(self):
        return self._gaze_dname

    @property
    def gaze_dpath(self):
        return join(self._gaze_data_root, self.gaze_dname)

    @property
    def scene_dname(self):
        return self._scene_dname

    @property
    def scene_dpath(self):
        return join(self._scene_data_root, self.scene_dname)

    def get_gaze_data_dpath(self, subject_id):

        return join(self.gaze_dpath, subject_id)

    def get_gaze_data_fpath(self, subject_id, xid):

        fname = "{subject_id}_{xid}.h5".format(subject_id=subject_id,
                                               xid=xid)

        return join(self.get_gaze_data_dpath(subject_id), fname)

    def get_scene_data_dpath(self, subject_id, xid):

        dname = '{subject_id}_{xid}'.format(subject_id=subject_id, xid=xid)

        return join(self.scene_dpath, subject_id, dname)

    def get_scene_data_fpaths(self, subject_id, xid):

        return glob(join(self.get_scene_data_dpath(subject_id, xid), '*.bmp'))


class RawDataManager(DataManager):

    def __init__(self):

        root = join(config.root_data_dpath, config.raw_dname)
        super(RawDataManager, self).__init__(root)

    def get_radials_dpath(self, subject_id, dist, rep):

        radials_dname = 'radials_{dist}_{rep}'.format(dist=dist, rep=rep)

        return join(self.get_scene_data_dpath(subject_id, session_id), radials_dname)

    def get_radials_fpaths(self, subject_id, session_id, dist, rep):

        return glob(self.get_radials_dpath(subject_id, session_id, dist, rep),
                    '*.bmp')


class ProcessedDataManager(DataManager):

    def __init__(self):

        root = join(config.root_data_dpath, config.processed_dname)
        super(ProcessedDataManager, self).__init__(root)