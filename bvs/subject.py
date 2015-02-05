"""
This module provides a class to hold subject metadata such as eye information
and manage input and output directories based on the configuration file.
"""

from __future__ import absolute_import, print_function, division, unicode_literals

import os.path

class BORISSubject(object):

    def __init__(self, subject_id):

        self._subject_id = subject_id

    @property
    def subject_id(self):
        """Returns the id code of the subject."""

        return self._subject_id

    @property
    def raw_gaze_dpath(self):
        return os.path.join(ROOT_DATA_DPATH, 'raw', 'gaze', self.subject_id)

    def raw_ascii_fpath(self, task_id):
        """Returns the file path to the raw ascii data for the task specified
        by task_id."""

        asc_fname = "{subject_id}_{task_id}.asc".format(subject_id=self.subject_id,
                                                        task_id=task_id)
        return os.path.join(self.raw_gaze_dpath, asc_fname)

    def needs_framesync(self, task_id):
        """Returns whether the subject needs manual framesyncing for the task
        specified by task_id."""

        framesync_fname = "{subject_id}_{task_id}.framesync".format(
                            subject_id=self.subject_id, task_id=task_id)

        if os.path.exists(os.path.join(self.raw_gaze_dpath, framesync_fname)):
            return True
        else:
            return False