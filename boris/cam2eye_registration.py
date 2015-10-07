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

    def __init__(self, data_fpaths, processed_data_fpath, stereocalibration_processed_fpath, trialnum):

        self.data_fpaths                        = data_fpaths    # path to directory with images
        self.processed_data_fpath               = join(processed_data_fpath,'trial_{tr}'.format(tr=str(trialnum)))    # path to directory to save results
        self.stereocalibration_processed_fpath  = stereocalibration_processed_fpath    # path to directory to save results
        self.trialnum                           = trialnum # capture trial num, 1 = before task, 2 = after task
        self._display   = False
        self._dims      = (4,11)    # number of circle row, columns


    @property
    def display(self):
        return self._display
     
    @property
    def dims(self):
        return self._dims

    def register(self):
        ''' perform and assess error of camera pose estimation relative to the cyclopean eyes
        '''

        # make directory for results and images
        if not os.path.exists(join(self.processed_data_fpath,'images_used')):
            os.makedirs(join(self.processed_data_fpath,'images_used'))


        ### DETECT AND STORE IMAGE POINTS ###
        print 'Detecting Circle Grid points...'
            
        image_coords, circle_images = self.find_circle_centers()
        world_coords                = self.get_circle_world_coords()
        
        self.load_camera_params()

        ### DETECT AND STORE IMAGE POINTS ###
        print 'Esimating Circle Grid pose...'
        retval, rvec, tvec = cv2.solvePnP(objectPoints=world_coords,
                                      imagePoints=image_coords,
                                      cameraMatrix=self.cam_mat,
                                      distCoeffs=self.dist_coeffs,
                                      useExtrinsicGuess=0)

        camera_coords, jac  = cv2.projectPoints(objectPoints=world_coords,rvec=rvec,tvec=tvec,cameraMatrix=self.cam_mat,distCoeffs=self.dist_coeffs)
        camera_coords       = np.reshape(camera_coords,(132,2))
        error               = np.sqrt(np.power(camera_coords[:,0]-image_coords[:,0],2) + np.power(camera_coords[:,1]-image_coords[:,1],2))
        rms                 = np.sqrt(np.mean(np.power(error,2)))

        print "Transformations found!"
        print "solvepnp rms error: " + str(rms)
        print "tvec: " + str(tvec)
        print "rvec: " + str(rvec*(180/np.pi))

        self.rvec = rvec
        self.tvec = tvec
        self.rms = rms

        return None


    def find_circle_centers(self):
        """Returns an array of circle grid coordinates in an image from each distance of 50cm, 100cm
        and 450cm in that order."""

        print "Finding circle centers..."
        image_coords    = []
        point_images   = []

        dist = [50, 100, 450];
        cnt = 0;

        for directory in self.data_fpaths:

            img_name    = 'cam1_frame_1.bmp'
            print "Finding circles in", os.path.join(directory, img_name)

            # load image and find circles
            board_image                 = cv2.imread(os.path.join(directory, img_name),1) 
            [pattern_was_found,points]  = cv2.findCirclesGridDefault(board_image,self._dims,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            points                      = points.squeeze()  # CHANGED: moved squeeze to before corner checking instead of after
            point_image                 = board_image.copy()

            # save image
            cv2.drawChessboardCorners(point_image, self._dims, points, pattern_was_found)
            fname = 'cam1_distance_{distance}cm.bmp'.format(distance=str(dist[cnt]))
            cv2.imwrite(join(self.processed_data_fpath,'images_used',fname),point_image)

            image_coords.append(points)
            point_images.append(point_image)

            cnt += 1

        return np.concatenate(image_coords), point_images

    def get_circle_world_coords(self):
        """
        This returns an array of the world coordinates (cyclopean origin) of the centers of the CIRCLES
        in the CIRCLEGRID. Pixel offsets were measured in photoshop and are therefor hard coded.
        This function is specific to the acircles_pattern_grey_DIST.png files used to display the
        circle grid in the experiment. If those images were not used on the screen, DO NOT USE THIS
        FUNCTION.
        """

        x_offset                = 597.5  # start 597.5 pixels in from the left side of the board png
        y_offset                = 69.50  # start 69.50 pixels down from the top side of the board png
        inter_circle_distance   = 52.25*2.0  # Pixel distance between circles in the same row
        center_points_pix_png   = []  # List to fill with png pixel positions
        shift_direction         = 1  # When we move to the right to calculate the next set of centers, we need to shift the y_offset in the negative direction

        # ATTENTION! Looking at the calibration frames, it looks like a (4,11) sized board starts with the LOWER LEFT circle and works its way up so that is what I am assuming!!
        size = (11,4)  # THIS SIZE IS NOT REFLECTIVE OF BOARD ORIENTATION! YES, I MEAN TO DO 11 X-POSITIONS AND 4 Y-POSITIONS TO CALCULATE THE WORLD COORDINATES!!
        for x_pos in xrange(size[0]):
            for y_pos in xrange(size[1]):
                next_point = (x_offset - x_pos*inter_circle_distance/2.0,  # Move over to the left half the inter-circle distance each time we want to
                              y_offset + y_pos*inter_circle_distance)  # Move down by the intercircle distance
                center_points_pix_png.append(next_point)

            y_offset += shift_direction * inter_circle_distance/2.0  # We need to change the y offset when we move to the next x position
            shift_direction *= -1  # Next time we'll want to shift the y_offset in the opposite direction

        center_points_pix_png = np.array(center_points_pix_png)  # Turn this into an array


        center_points_cm_world = np.zeros((size[0]*size[1]*3, 3), dtype='float')
        start_idx = 0
        end_idx = size[0] * size[1]
        for dist in [50, 100, 450]:
            if dist == 50:
                scale = .5
                top_left = ((1920-337)/2.0, (1024-250)/2.0)  # top left corner of the 50 png in pixels with (0,0) at center, minus y is UP
            elif dist == 100:
                scale = 1.0
                top_left = ((1920-674)/2.0, (1024-500)/2.0)
            elif dist == 450:
                scale = 2.0
                top_left = ((1920-1349)/2.0, (1024-1000)/2.0)

            center_points_pix_screen = top_left + (center_points_pix_png * scale)  # shift the grid to be in the center of the screen and scale them
        
            center_points_pix_screen -= (1920/2.0, 1024/2.0)
            #print "circle centers in screen pixels:\n", center_points_pix_screen

            cm_per_pix_x = 121.0/1920  # LG 3D TV width in cm over pixels wide
            cm_per_pix_y = 68.0/1024
            center_points_cm_world[start_idx:end_idx, 0] = cm_per_pix_x * center_points_pix_screen[:,0]  # Convert x position to cm
            center_points_cm_world[start_idx:end_idx, 1] = cm_per_pix_y * -center_points_pix_screen[:,1]  # Convert y position to cm, remember POSITIVE Y is now UP!
            center_points_cm_world[start_idx:end_idx, 2] = float(dist)

            start_idx += size[0]*size[1]
            end_idx += size[0]*size[1]

        #print "circle centers in cm:\n", center_points_cm_world

        return center_points_cm_world

    def load_camera_params(self):
        ''' load distortion coefficients and intrinsics for camera that captured circle grid
        '''

        self.dist_coeffs = np.loadtxt(join(self.stereocalibration_processed_fpath, "Distortion_cam1.txt"), delimiter=None, ndmin=0)
        self.cam_mat     = np.loadtxt(join(self.stereocalibration_processed_fpath, "Intrinsics_cam1.txt"), delimiter=None, ndmin=2)

        return None

    def store_params(self):
        ''' store results '''

        np.savez(join(self.processed_data_fpath,'Cam1_pose_and_intrinsics.npz'), rvec=self.rvec, tvec=self.tvec, cam_mat=self.cam_mat, dist_coeffs=self.dist_coeffs)
        np.savetxt(join(self.processed_data_fpath,'Intrinsics_cam1_copy.txt'), self.cam_mat)
        np.savetxt(join(self.processed_data_fpath,'Distortion_cam1_copy.txt'), self.dist_coeffs)
        np.savetxt(join(self.processed_data_fpath,'WorldRotation_cam1.txt'), self.rvec)
        np.savetxt(join(self.processed_data_fpath,'WorldTranslation_cam1.txt'), self.tvec)

        calib_params = open(join(self.processed_data_fpath,'Calibration_quality.txt'), 'w')
        calib_params.write("\n rms: " + str(self.rms))
        calib_params.close()

        return None


if __name__ == '__main__':
    import os

    frames_dpath = '../../session_data/raw/cam2eye_registration/kre/kre_cafe/'
    processed_dpath = '../../session_data/processed/cam2eye_registration/kre/kre_cafe/'
    stereo_processed_dpath = '../../session_data/processed/stereocalibration/kre/kre_cafe/'

    cam2eye_register = CamEyeRegister(frames_dpath,processed_dpath,stereo_processed_dpath,1)
    cam2eye_register.register()

    #save parameter estimates
    #print("\nSaving all parameters to the folder with checkerboard images...")
    #stereo_calibrator.store_calib_params()

