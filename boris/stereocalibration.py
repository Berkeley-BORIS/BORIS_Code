import sys
import os
import random
import fnmatch
import cv2
import numpy as np

if cv2.__version__.startswith('2.3'):
    raise NotImplementedError("WARNING: cv2 is version {0}, Z axis is inverted in this version and still result in incorrect results".format(cv2.__version__))

## NOTES: I removed the error for 2.4 and replaced it with an error for 2.3, 
## since I think we should update everything to work in the current coordinate system (Z positive?)
## Currently results are printed to terminal and savd to txt files, need to updat this
## for however we want to do it in BORIS

class StereoCalibrator(object):
    '''reads files from a directory of stereo images of opencv circle grid. calibrates intrinsics of each camera, then extrinsics of stereo rig
    '''

    def __init__(self, data_fpath):

        self.data_fpath = data_fpath    # path to director with images
        self._nimages   = 100       # number of images to load
        self._display   = False
        self._dims      = (4,11)    # number of circle row, columns
        self._size      = (640,480) # image size
        self._num_pts   = self._dims[0] * self._dims[1]     # number of circles in grid

    @property
    def nimages(self):
        return self._nimages

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

    def calibrate(self):

        ### DETECT AND STORE IMAGE POINTS ###
        print 'Detecting Circle Grid points...'
            
        # make directory for storing the specific images we use
        if not os.path.exists(os.join(self.data_fpath,'images_used')):
            os.mkdir(os.join(self.data_fpath,'images_used'))

        nimg    = 0     # number of images with found corners
        iptsF1  = []    # image point arrays to fill up
        iptsF2  = []
        image_inds = range(1,500,5) + range(2,500,5) + range(3,500,5) + range(4,500,5) + range(5,500,5) #order in which to grab images

        # for each image number
        for n in image_inds:

            # filenames for each camera
            filename1 = os.join(self.data_fpath,'cam1_frame_'+str(n)+'.bmp')
            filename2 = os.join(self.data_fpath,'cam2_frame_'+str(n)+'.bmp')

            # if both files exist
            if os.path.exists(filename1) and os.path.exists(filename2):

                # read in the images and look for the circle grid points
                [img1,found1,points1] = self.get_circle_points(filename1)
                [img2,found2,points2] = self.get_circle_points(filename2)

                # if the circle grid points were located in both camera images
                if found1 and found2:

                    # copy the found points into the ipts matrices
                    iptsF1.append(np.reshape(points1,(num_pts,2)))
                    iptsF2.append(np.reshape(points2,(num_pts,2)))

                    #save images with points identified 
                    self.save_image_with_grid_drawn(n,img1,points1,found1)
                    self.save_image_with_grid_drawn(n,img2,points2,found2)

                    nimg = nimg + 1 #increment image counter
                    # check if you've reached the target number of images, otherwise continue
                    if nimg == nimages:
                        break
                    
                
        print "\n Usable stereo pairs: " + str(nimg)

        if 0 :
            # convert image points to numpy
            iptsF1 = np.array(iptsF1, dtype = np.float32)
            iptsF2 = np.array(iptsF2, dtype = np.float32)


            ### PERFORM SINGLE CAMERA CALIBRATION ###

            # evaluate object points
            opts = objectpoints(dims,nimg,4.35)

            # initialize BORIS camera parameters
            [intrinsics1,distortion1] = initialize_camera_matrices()
            [intrinsics2,distortion2] = initialize_camera_matrices()


            #calibrate cameras
            print 'Calibrating camera 1...'
            (cam1rms, intrinsics1, distortion1, rotv1, trav1) = cv2.calibrateCamera(opts, iptsF1, size, intrinsics1, distortion1, flags=int(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL))
            report_camera_matrices(intrinsics1,distortion1)
            
            print 'Calibrating camera 2...'
            (cam2rms, intrinsics2, distortion2, rotv2, trav2) = cv2.calibrateCamera(opts, iptsF2, size, intrinsics2, distortion2,flags=int(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL))
            report_camera_matrices(intrinsics2,distortion2)

            print "\n rms pixel error:\ncam1 orig: " + str(cam1rms),"\ncam2 orig: " + str(cam2rms)


            ### PERFORM STEREO CALIBRATION ###

            # Estimate extrinsic parameters from stereo point correspondences
            print "\n Stereo estimating..."
            (stereorms, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F) = cv2.stereoCalibrate(opts, iptsF1, iptsF2, size, intrinsics1, distortion1, intrinsics2, distortion2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL))
           
            print "\nEstimated extrinsic parameters between cameras 1 and 2:\nRotation:",R
            print "\nTranslation:",[T[i,0] for i in range(3)]
            print "\n rms pixel error:\nstereo: " + str(stereorms)


            # Initialize empty rectification parameters
            (R1,R2,P1,P2,Q) = initialize_rect_params()

            # Estimate rectification
            cv2.stereoRectify(intrinsics1, distortion1, intrinsics2, distortion2, size, R,T,R1, R2, P1,P2, Q,cv2.CALIB_ZERO_DISPARITY)

            # get rectification maps
            (map1x,map1y) = cv2.initUndistortRectifyMap(intrinsics1, distortion1, R1, P1, (640,480), cv2.CV_32FC1)
            (map2x,map2y) = cv2.initUndistortRectifyMap(intrinsics2, distortion2, R2, P2, (640,480), cv2.CV_32FC1)
            

            ### THIS PROBABLY CHANGES WITH BORIS ###

            #save parameter estimates
            print "\nSaving all parameters to the folder with checkerboard images..."
            store_calib_params(check_img_folder,nimg,cam1rms,cam2rms,stereorms,intrinsics1,intrinsics2,distortion1,distortion2,P1,P2,E,F,R,T,Q,R1,R2,map1x,map1y,map2x,map2y)

        return None



    def get_circle_points(self,filename):
        ''' load in an image file and return the image pixel coordinates of circle grid points
        '''

        # read in image
        img = cv2.imread(filename,0)

        # find center points in circle grid
        [found,points] = cv2.findCirclesGridDefault(img,self._dims,flags=(cv2.CALIB_CB_ASYMMETRIC_GRID))

        return img,found,points

                    
    def save_image_with_grid_drawn(self,n,img,points,found):
        '''save a copy of the image with circle grid points identified  
        '''
                    
        drawn_boards = img.copy()
        cv2.drawChessboardCorners(drawn_boards, self._dims, points, found)
        cv2.imwrite(os.join(self.data_fpath,'images_used',{'cam1_frame_',str(n+1),'.bmp'}, drawn_boards)

        return None

    def objectpoints(dims,num_images,square_size):
        '''determine 3d object points for each image
        '''
        
        # circle grid dimensions
        width   = dims[0]
        height  = dims[1]
        num_pts = width*height

        # initialize emtpy object point array
        opts = []

        # for each image
        for n in range(num_images):

            # make zero array for coordinates
            temp = np.zeros( (num_pts,3) )

            # for each point on the circle grid
            for i in range(height):
                for j in range(width):

                        # determine xy location on the grid (z is always zero)
                        if i%2==0:
                            temp[i*width+j,0] = (i*(square_size/2.00))
                            temp[i*width+j,1] = j*square_size
                            temp[i*width+j,2] = 0
                        else:
                            temp[i*width+j,0] = (i*(square_size/2.00))
                            temp[i*width+j,1] = (j*square_size) + square_size/2.00
                            temp[i*width+j,2] = 0

            # add to object points array                
            opts.append(temp)

        # make numpy    
        opts = np.array(opts, dtype = np.float32)

        return opts

    def initialize_camera_matrices():
        '''initial estimates for the camera intrinsics and distortion
        '''

        intrinsics = np.zeros( (3,3) )
        distortion = np.zeros( (8,1) )

        # Set initial guess for intrinsic camera parameters (focal length = 0.35cm)
        intrinsics[0,0] = 583.3 
        intrinsics[1,1] = 583.3 
        intrinsics[0,2] = 320
        intrinsics[1,2] = 240
        intrinsics[2,2] = 1.0

        return intrinsics,distortion

    def report_camera_matrices(intrinsics,distortion):
        '''just print out camera intrinsics results
        '''

        print "\nEstimated intrinsic parameters:\n",intrinsics
        print "\nEstimated distortion parameters:\n",distortion

        return None 

            
    def initialize_rect_params():
        '''Initialize empty stereo rectification parameters
        '''

        R1  = np.zeros( (3,3) )
        R2  = np.zeros( (3,3) )
        P1  = np.zeros( (3,4) )
        P2  = np.zeros( (3,4) )
        Q   = np.zeros( (4,4) )

        return R1,R2,P1,P2,Q

    def store_calib_params(check_img_folder,nimg,cam1rms,cam2rms,stereorms,intrinsics1,intrinsics2,distortion1,distortion2,P1,P2,E,F,R,T,Q,R1,R2,map1x,map1y,map2x,map2y):
        '''write calibration parameters to text and XML files
        '''

        calib_params = open(check_img_folder+ 'calib_params.txt', 'w')
        calib_params.write("\n num stereo frames: " + str(nimg))
        calib_params.write("\n rms cam1: " + str(cam1rms))
        calib_params.write("\n rms cam2: " + str(cam2rms))
        calib_params.write("\n rms stereo: " + str(stereorms))
        calib_params.close()

        np.savetxt(check_img_folder + 'Intrinsics_cam1.txt', intrinsics1)
        np.savetxt(check_img_folder + 'Intrinsics_cam2.txt', intrinsics2)

        np.savetxt(check_img_folder + 'Distortion_cam1.txt', distortion1)
        np.savetxt(check_img_folder + 'Distortion_cam2.txt', distortion2)

        np.savetxt(check_img_folder + 'Projection_matrix_cam1.txt',P1)
        np.savetxt(check_img_folder + 'Projection_matrix_cam2.txt',P2)
        np.savetxt(check_img_folder + 'Essential_matrix.txt',E)
        np.savetxt(check_img_folder + 'Fundamental_matrix.txt',F)
        np.savetxt(check_img_folder + 'Rotation_matrix.txt',R)
        np.savetxt(check_img_folder + 'Translation_vector.txt',T)
        np.savetxt(check_img_folder + 'Disp2depth_matrix.txt',Q)

        np.savetxt(check_img_folder + 'Rectification_transform_cam1.txt',R1)
        np.savetxt(check_img_folder + 'Rectification_transform_cam2.txt',R2)
        np.savetxt(check_img_folder + 'Rectification_map_cam1x.txt',map1x)
        np.savetxt(check_img_folder + 'Rectification_map_cam1y.txt',map1y)
        np.savetxt(check_img_folder + 'Rectification_map_cam2x.txt',map2x)
        np.savetxt(check_img_folder + 'Rectification_map_cam2y.txt',map2y)




if __name__=="__main__":
    check_img_folder = sys.argv[1]
    nimages = int(sys.argv[2])

    stereocalibration(check_img_folder,nimages=100,display=False,dims=(4,11),size=(640,480))

