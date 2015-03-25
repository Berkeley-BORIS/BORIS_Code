import sys
import os
import random
import fnmatch
import cv
import cv2
import numpy as np

print cv2.__version__
if not cv2.__version__.startswith('2.3'):
	raise NotImplementedError("WARNING: cv2 is version {0}!! We haven't implemented the inverted transform direction changed after 2.3!".format(cv2.__version__))

print asdf

def stereo_calibration(check_img_folder,nimages,display=False,dims=(4,11),size=(640,480)):
	'''reads files from a directory of stereo images of opencv circle grid. calibrates intrinsics of each camera, then extrinsics of stereo rig
	'''
	
	#grab calbration frames directory
	for dir in os.listdir(check_img_folder):
		if fnmatch.fnmatch(dir,'calibration_frames*'):
			check_img_folder = check_img_folder + dir + '/'
			break
			
    # Number of points in circle grid
	num_pts = dims[0] * dims[1]

	if not os.path.exists(check_img_folder + 'images_used/'):
		os.mkdir(check_img_folder + 'images_used/')

	# evaluate image points
	nimg = 0 #number of images with found corners
	iptsF1 = [] #image point arrays to fill up
	iptsF2 = []
	random_images = random.sample(range(500), nimages)
	#for n in range(0,nimages,2):
	for n in random_images:
		filename1 = check_img_folder + 'cam1_frame_'+str(n+1)+'.bmp'
		filename2 = check_img_folder + 'cam2_frame_'+str(n+1)+'.bmp'
		if os.path.exists(filename1) and os.path.exists(filename2):
			img1 = cv2.imread(filename1,0)
			img2 = cv2.imread(filename2,0)
			# find center points in circle grid
			[found1,points1] = cv2.findCirclesGridDefault(img1,dims,flags=(cv2.CALIB_CB_ASYMMETRIC_GRID))
			[found2,points2] = cv2.findCirclesGridDefault(img2,dims,flags=(cv2.CALIB_CB_ASYMMETRIC_GRID))
			# copy the found points into the ipts matrices
			temp1 = np.zeros( (num_pts,2) )
			temp2 = np.zeros( (num_pts,2) )
			if found1 and found2:
				for i in range(num_pts):
					temp1[i,0]=points1[i,0,0]
					temp1[i,1]=points1[i,0,1]
					temp2[i,0]=points2[i,0,0]
					temp2[i,1]=points2[i,0,1]

				iptsF1.append(temp1)
				iptsF2.append(temp2)
				nimg = nimg + 1 #increment image counter
					
				#save images with points identified	
				drawn_boards_1 = img1.copy()
				drawn_boards_2 = img2.copy()
				cv2.drawChessboardCorners(drawn_boards_1, dims, points1, found1)
				cv2.drawChessboardCorners(drawn_boards_2, dims, points2, found2)
				cv2.imwrite(check_img_folder + 'images_used/' +  'cam1_frame_'+str(n+1)+'.bmp', drawn_boards_1)
				cv2.imwrite(check_img_folder + 'images_used/' +  'cam2_frame_'+str(n+1)+'.bmp', drawn_boards_2)

				
	print "\n Usable stereo pairs: " + str(nimg)
	
	# convert image points to numpy
	iptsF1 = np.array(iptsF1, dtype = np.float32)
	iptsF2 = np.array(iptsF2, dtype = np.float32)

	# evaluate object points
	opts = object_points(dims,nimg,4.35)

	# initialize camera parameters
	intrinsics1 = np.zeros( (3,3) )
	intrinsics2 = np.zeros( (3,3) )
	distortion1 = np.zeros( (8,1) )
	distortion2 = np.zeros( (8,1) )

	# Set initial guess for intrinsic camera parameters (focal length = 0.35cm)
	intrinsics1[0,0] = 583.3 
	intrinsics1[1,1] = 583.3 
	intrinsics1[0,2] = 320
	intrinsics1[1,2] = 240
	intrinsics1[2,2] = 1.0

	intrinsics2[0,0] = 583.3 
	intrinsics2[1,1] = 583.3 
	intrinsics2[0,2] = 320
	intrinsics2[1,2] = 240
	intrinsics2[2,2] = 1.0

	#calibrate cameras
	print 'Calibrating camera 1...'
	(cam1rms, intrinsics1, distortion1, rotv1, trav1) = cv2.calibrateCamera(opts, iptsF1, size, intrinsics1, distortion1, flags=int(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL))
	print "\nEstimated intrinsic parameters for camera 1:"
	for i in range(3):
		print [intrinsics1[i,j] for j in range(3)]
   
	print "\nEstimated distortion parameters for camera 1:"
	print distortion1	
	
	print 'Calibrating camera 2...'
	(cam2rms, intrinsics2, distortion2, rotv2, trav2) = cv2.calibrateCamera(opts, iptsF2, size, intrinsics2, distortion2,flags=int(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL))
	
	print "\nEstimated intrinsic parameters for camera 2:"
	for i in range(3):
		print [intrinsics2[i,j] for j in range(3)]

	print "\nEstimated distortion parameters for camera 2:"
	print distortion2

	print "\n rms pixel error:"
	print "cam1 orig: " + str(cam1rms)
	print "cam2 orig: " + str(cam2rms)
	
	# Estimate extrinsic parameters from stereo point correspondences
	print "\n Stereo estimating..."
	#(stereorms, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F) = cv2.stereoCalibrate(opts, iptsF1, iptsF2, intrinsics1, distortion1, intrinsics2, distortion2, size,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL))
	(stereorms, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F) = cv2.stereoCalibrate(opts, iptsF1, iptsF2, size, intrinsics1, distortion1, intrinsics2, distortion2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL))
   
	print "\nEstimated extrinsic parameters between cameras 1 and 2:\nRotation:"
	for i in range(3):
		print [R[i,j] for j in range(3)]

	print "\nTranslation:"
	print [T[i,0] for i in range(3)]

	print "\n rms pixel error:"
	print "stereo: " + str(stereorms)


    # Initialize rectification parameters
	R1=cv.CreateMat(3,3,cv.CV_64F)
	R2=cv.CreateMat(3,3,cv.CV_64F)
	P1=cv.CreateMat(3,4,cv.CV_64F)
	P2=cv.CreateMat(3,4,cv.CV_64F)
	Q=cv.CreateMat(4,4,cv.CV_64F)

	intrinsics1 = cv.fromarray(intrinsics1.copy())
	intrinsics2 = cv.fromarray(intrinsics2.copy())
	distortion1 = cv.fromarray(distortion1.copy())
	distortion2 = cv.fromarray(distortion2.copy())
	R = cv.fromarray(R.copy())
	T = cv.fromarray(T.copy())
	E = cv.fromarray(E.copy())
	F = cv.fromarray(F.copy())
	new_size = (640,480)

	# Estimate rectification
	(roi1,roi2)=cv.StereoRectify(intrinsics1, intrinsics2, distortion1, distortion2, size, R,T,R1, R2, P1,P2, Q,cv.CV_CALIB_ZERO_DISPARITY)

	# Rectification maps
	#Left maps
	map1x = cv.CreateMat(new_size[1], new_size[0], cv.CV_32FC1)
	map2x = cv.CreateMat(new_size[1], new_size[0], cv.CV_32FC1)

	#Right maps
	map1y = cv.CreateMat(new_size[1], new_size[0], cv.CV_32FC1)
	map2y = cv.CreateMat(new_size[1], new_size[0], cv.CV_32FC1)

	cv.InitUndistortRectifyMap(intrinsics1, distortion1, R1, P1, map1x, map1y)
	cv.InitUndistortRectifyMap(intrinsics2, distortion2, R2, P2, map2x, map2y)

	#save parameter estimates
	print "\nSaving all parameters to the folder with checkerboard images..."
	calib_params = open(check_img_folder+ 'calib_params.txt', 'w')
	calib_params.write("\n num stereo frames: " + str(nimg))
	calib_params.write("\n rms cam1: " + str(cam1rms))
	calib_params.write("\n rms cam2: " + str(cam2rms))
	calib_params.write("\n rms stereo: " + str(stereorms))
	calib_params.close()
	
	cv.Save(check_img_folder + 'Intrinsics_cam1.xml',intrinsics1)
	cv.Save(check_img_folder + 'Intrinsics_cam2.xml',intrinsics2)
	cv.Save(check_img_folder + 'Distortion_cam1.xml',distortion1)
	cv.Save(check_img_folder + 'Distortion_cam2.xml',distortion2)
	cv.Save(check_img_folder + 'Projection_matrix_cam1.xml',P1)
	cv.Save(check_img_folder + 'Projection_matrix_cam2.xml',P2)
	cv.Save(check_img_folder + 'Essential_matrix.xml',E)
	cv.Save(check_img_folder + 'Fundamental_matrix.xml',F)
	cv.Save(check_img_folder + 'Rotation_matrix.xml',R)
	cv.Save(check_img_folder + 'Translation_vector.xml',T)
	cv.Save(check_img_folder + 'Disp2depth_matrix.xml',Q)

	cv.Save(check_img_folder + 'Rectification_transform_cam1.xml',R1)
	cv.Save(check_img_folder + 'Rectification_transform_cam2.xml',R2)
	cv.Save(check_img_folder + 'Rectification_map_cam1x.xml',map1x)
	cv.Save(check_img_folder + 'Rectification_map_cam1y.xml',map1y)
	cv.Save(check_img_folder + 'Rectification_map_cam2x.xml',map2x)
	cv.Save(check_img_folder + 'Rectification_map_cam2y.xml',map2y)

	return None

def object_points(dims,num_images,square_size):
	'''determine 3d object points for each image
	'''
	
	width = dims[0]
	height = dims[1]
	num_pts = width*height

	opts = []
	for n in range(num_images):
		temp = np.zeros( (num_pts,3) )
		for i in range(height):
			for j in range(width):
					if i%2==0:
						temp[i*width+j,0] = (i*(square_size/2.00))
						temp[i*width+j,1] = j*square_size
						temp[i*width+j,2] = 0
					else:
						temp[i*width+j,0] = (i*(square_size/2.00))
						temp[i*width+j,1] = (j*square_size) + square_size/2.00
						temp[i*width+j,2] = 0
		opts.append(temp)
	opts = np.array(opts, dtype = np.float32)

	return opts
	
if __name__=="__main__":
    check_img_folder = sys.argv[1]
    nimages = int(sys.argv[2])

    stereo_calibration(check_img_folder,nimages,display=False,dims=(4,11),size=(640,480))

