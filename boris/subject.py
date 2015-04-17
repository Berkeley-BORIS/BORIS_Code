"""
This module provides a class to hold subject metadata such as eye information
and manage input and output directories based on the configuration file.
"""

from __future__ import absolute_import, print_function, division, unicode_literals

from os.path import join, exists

import yaml

from . import config


class BORISSubject(object):

    def __init__(self, subject_id):

        self._subject_id = subject_id
        self._load_eyeinfo()

    @property
    def subject_id(self):
        """Returns the id code of the subject."""

        return self._subject_id

    @property
    def raw_gaze_dpath(self):
        return join(config.raw_gaze_dpath, self.subject_id)

    @property
    def processed_gaze_dpath(self):
        return join(config.processed_gaze_dpath, self.subject_id)

    @property
    def ipd(self):

        return self._eyeinfo['ipd']

    @property
    def pupil_size(self):

        return self._eyeinfo['pupil']

    def raw_ascii_fpath(self, session_id):
        """Returns the file path to the raw ascii data for the task specified
        by session_id."""

        asc_fname = "{subject_id}_{session_id}.asc".format(subject_id=self.subject_id,
                                                        session_id=session_id)
        return join(self.raw_gaze_dpath, asc_fname)

    def needs_framesync(self, session_id):
        """Returns whether the subject needs manual framesyncing for the task
        specified by session_id."""

        framesync_fname = "{subject_id}_{session_id}.framesync".format(
                            subject_id=self.subject_id, session_id=session_id)

        if exists(join(self.raw_gaze_dpath, framesync_fname)):
            return True
        else:
            return False

    def framesync_fpath(self, session_id):
        """Returns the file path the framesync file."""

        framesync_fname = "{subject_id}_{session_id}.framesync".format(
                            subject_id=self.subject_id, session_id=session_id)

        return join(self.raw_gaze_dpath, framesync_fname)

    def gaze_data_fpath(self, session_id):
        """Returns the file path to the gaze data file for this subject and task.
        """

        task_fname = "{subject_id}_{session_id}.h5".format(
                      subject_id=self.subject_id, session_id=session_id)

        return join(self.processed_gaze_dpath, task_fname)

    def _load_eyeinfo(self):

        eyeinfo_fname = "{subject_id}.eyeinfo".format(subject_id=self.subject_id)
        with open(join(self.raw_gaze_dpath, eyeinfo_fname), 'r') as f:
            self._eyeinfo = yaml.load(f)