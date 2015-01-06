"""
Testing module for eyeproc.
"""
import pytest
import numpy as np

import eyeproc

@pytest.fixture
def z_dir_backwards():
    z_dir = np.array([[0., 0, 1]]).T

    return z_dir

@pytest.fixture
def other_vec():
    z_dir = np.array([0., 0, 1])
    angles = [5, 20, 45, 80]
    rotated_vecs = []
    for angle in angles:
        R = eyeproc.get_R(0, angle)
        rotated_vecs.append(np.dot(R, z_dir))
    rotated_vecs = np.r_[rotated_vecs]
    return rotated_vecs, angles

@pytest.fixture(params=[(3), (1,3), (5,3)])
def properly_shaped_vec(request):
    vec = np.array([0,0,1.])

    new_shape = request.param
    vec = np.resize(vec, new_shape)

    return vec

@pytest.fixture
def z_unit_vec():

    return np.array([0,0,1.])

@pytest.fixture(params=[(3,1), (3, 5)])
def wrong_shaped_vec(request):

    vec = np.array([0,0,1.])

    new_shape = request.param
    vec = np.resize(vec, new_shape)

    return vec


def rotate_by_angle(vec, angle):

    if vec.ndim == 1:
        vec = np.expand_dims(vec, 1)

    if not hasattr(angle, '__iter__'):
        angle = [angle]

    rotated_vec = []
    for a in angle:
        rotated_vec.append(np.dot(eyeproc.get_R(0, a), vec).T)

    rotated_vec = np.concatenate(rotated_vec, 0)

    return rotated_vec

@pytest.fixture(params=[5, (5, 10)])
def angle(request):
    return request.param

class TestCalcAngle(object):

    calc_angle = staticmethod(eyeproc.calc_angle)

    def test_should_accept_shapes(self, properly_shaped_vec):
        angle = self.calc_angle(properly_shaped_vec, properly_shaped_vec)
        assert (angle == 0).all()

    def test_should_raise_on_wrong_shapes(self, wrong_shaped_vec, z_unit_vec):
        with pytest.raises(AssertionError):
            self.calc_angle(z_unit_vec, wrong_shaped_vec)
            self.calc_angle(wrong_shaped_vec, z_unit_vec)
            self.calc_angle(wrong_shaped_vec, wrong_shaped_vec)

    def test_returns_correct_angle(self, angle):

        vec_1 = np.array([0., 0., 1.])
        vec_2 = rotate_by_angle(vec_1, angle)

        assert np.allclose(self.calc_angle(vec_1, vec_2), angle)
        assert np.allclose(self.calc_angle(vec_2, vec_1) , angle)


@pytest.fixture
def fixations_with_nans():

    fixations = np.array([[0, 0, 100.],
                          [np.nan, np.nan, np.nan],
                          [20, 0, 200],
                          [np.nan, 20, 1000]])

    return fixations

@pytest.fixture
def vergence():
    return 10

@pytest.fixture(params=np.linspace(5,8,10).tolist())
def ipd(request):
    return request.param

class TestVergence(object):

    calc_vergence = staticmethod(eyeproc.calc_vergence)

    def test_returns_correct_vergence_from_straight_ahead(self, vergence, ipd):


        dist = ipd / (2*np.tan(vergence/2. * np.pi/180))
        try:
            fixation_pt = np.zeros((len(dist),3))
        except TypeError:
            fixation_pt = np.zeros((1, 3))

        fixation_pt[:,2] = dist
        returned_vergence = self.calc_vergence(fixation_pt, ipd)
        assert np.allclose(returned_vergence, vergence).all()

    def test_should_accept_shapes(self, properly_shaped_vec):
        assert self.calc_vergence(properly_shaped_vec, ipd=6.5).any()

    def test_should_raise_on_wrong_shapes(self, wrong_shaped_vec):
        with pytest.raises(AssertionError):
            self.calc_vergence(wrong_shaped_vec, ipd=6.5)

    def test_returns_nans_at_missing_data(self, fixations_with_nans):
        rows_with_nans, = np.where(np.isnan(fixations_with_nans).any(axis=1))

        vergences = self.calc_vergence(fixations_with_nans, ipd=6.5)
        vergences_with_nans, = np.where(np.isnan(vergences))
        assert (vergences_with_nans == rows_with_nans).all()

    def test_returns_no_nans_at_full_data(self, fixations_with_nans):
        rows_without_nans, = np.where(~np.isnan(fixations_with_nans).any(axis=1))

        vergences = self.calc_vergence(fixations_with_nans, ipd=6.5)
        vergences_without_nans, = np.where(~np.isnan(vergences))
        assert (vergences_without_nans == rows_without_nans).all()

    def test_same_vergence_with_ints(self):
        ipd = 6
        fixation_pt = np.array([0,0,100])
        expected_vergence = self.calc_vergence(fixation_pt.astype(float), float(ipd))
        vergence = self.calc_vergence(fixation_pt, ipd)
        assert np.allclose(vergence, expected_vergence)

class TestVersion(object):

    calc_version = staticmethod(eyeproc.calc_version)

    def test_return_correct_version(self):

        expected_horz_versions = range(-18,18)
        expected_vert_versions = range(-18,18)
        expected_versions = [(h, v) for h in expected_horz_versions for v in expected_vert_versions]

        z_dir = np.array([0,0,100.])
        fixation_pt = []
        for v in expected_versions:
            R = eyeproc.get_R(*v)
            fixation_pt.append(np.dot(R,z_dir))

        fixation_pt = np.r_[fixation_pt]
        horz_versions, vert_versions = self.calc_version(fixation_pt)
        expected_horz_versions = np.array([ h for h, v in expected_versions])
        expected_vert_versions = np.array([v for h, v in expected_versions])
        assert (horz_versions - expected_horz_versions < 1e-5).all()
        assert (vert_versions - expected_vert_versions < 1e-5).all()

    def test_should_accept_shapes(self, properly_shaped_vec):
        assert self.calc_version(properly_shaped_vec)

    def test_should_raise_on_wrong_shapes(self, wrong_shaped_vec):
        with pytest.raises(AssertionError):
            self.calc_version(wrong_shaped_vec)

    def test_returns_nans_at_missing_data(self, fixations_with_nans):
        rows_with_nans, = np.where(np.isnan(fixations_with_nans).any(axis=1))

        horz_versions, vert_versions = self.calc_version(fixations_with_nans)
        for version in (horz_versions, vert_versions):
            versions_with_nans, = np.where(np.isnan(version))
            assert (versions_with_nans == rows_with_nans).all()

    def test_returns_no_nans_at_full_data(self, fixations_with_nans):
        rows_without_nans, = np.where(~np.isnan(fixations_with_nans).any(axis=1))

        versions = self.calc_version(fixations_with_nans)
        for version in versions:
            versions_without_nans, = np.where(~np.isnan(version))
            assert (versions_without_nans == rows_without_nans).all()

    def test_same_version_with_ints(self):
        fixation_pt = np.array([0,0,100])
        expected_version = self.calc_version(fixation_pt.astype(float))
        version = self.calc_version(fixation_pt)
        assert np.allclose(version, expected_version)