import os
from os.path import join
import random
import fnmatch
import cv2
import numpy as np

if cv2.__version__.startswith('2.3'):
    raise NotImplementedError("WARNING: cv2 is version {0}, Z axis is inverted in this version and still result in incorrect results".format(cv2.__version__))

## NOTE: I removed the error for 2.4 and replaced it with an error for 2.3, 
## since I think we should update everything to work in the current coordinate system (Z positive?)
## Currently results are printed to terminal and savd to txt files, need to updat this
## for however we want to do it in BORIS

class CamEyeRegister(object):
    '''reads files from a directory of circle grid images. estimate rotation and translation from camera to eyes
    '''

    def __init__(self, data_fpaths, processed_data_fpath):

        self.data_fpaths = data_fpaths    # path to directory with images
        self.processed_data_fpath = processed_data_fpath    # path to directory to save results
        self._display   = False
        self._dims      = (4,11)    # number of circle row, columns
        #self._size      = (640,480) # image size
        #self._num_pts   = self._dims[0] * self._dims[1]     # number of circles in grid


    @property
    def display(self):
        return self._display
     
    @property
    def dims(self):
        return self._dims

    @property
    def size(self):
        return self._size

    @property
    def num_pts(self):
        return self._num_pts

    def register(self):

        ### DETECT AND STORE IMAGE POINTS ###
        print 'Detecting Circle Grid points...'
            
        image_coords, circle_images = self.find_circle_centers()


    def find_circle_centers(self):
        """Returns an array of chessboard corners in an image from each distance of 50cm, 100cm
        and 450cm in that order."""

        print "Finding circle centers..."
        image_coords    = []
        point_images   = []

        for directory in self.data_fpaths:

            img_name    = 'cam1_frame_1.bmp'
            print "Finding circles in", os.path.join(directory, img_name)

            # load image and find circles
            board_image                 = cv2.imread(os.path.join(directory, img_name),1) 
            [pattern_was_found,points = cv2.findCirclesGridDefault(board_image,self._dims,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            points                     = points.squeeze()  # CHANGED: moved squeeze to before corner checking instead of after

            point_image = board_image.copy()
            cv2.drawChessboardCorners(point_image, self._dims, points, pattern_was_found)
            cv2.imwrite('./tmp.png',point_image)
            image_coords.append(points)
            point_images.append(point_image)

        return np.concatenate(image_coords), point_images


if __name__ == '__main__':
    import os

    frames_dpath = '../../session_data/raw/stereocalibration/kre/kre_cafe/calibration_frames_2012-08-01/'
    processed_dpath = '../../session_data/processed/stereocalibration/kre/kre_cafe/'

    stereo_calibrator = StereoCalibrator(frames_dpath,processed_dpath)
    stereo_calibrator.calibrate()

    #save parameter estimates
    #print("\nSaving all parameters to the folder with checkerboard images...")
    #stereo_calibrator.store_calib_params()

