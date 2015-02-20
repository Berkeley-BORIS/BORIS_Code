Steps to process disparity statistics data:

COPY DATA FROM DRIVE:
takes ~30 mins
(1) place stereo calibration directory in cameras/stereo_calibration/data/SUBJ/
(2) place eyelink ASC (SUBJ.asc) into eyes/data_processing/data/SUBJ
(3) place data from stereo cameras (SUBJ_images) in cameras/raw_data/

	note, if this is a nature walk or cafe task, there are probably about 10,000 frames of black or white. move these frames to a task_black directory

(4) in eyes/ipds create a text file called XXX.txt (subject's 3 initials) and enter ipd in cm


STEREO CAMERA CALIBRATION:
takes ~20 mins
in cameras/stereo_calibration/scripts, run calibration script on 250+ frames. Make sure yield of usable frames is about 150.
Rms stereo error should be <0.25 pixels
	
	example call: python stereo_calibration.py '../data/SUBJ/' 250


EYE DATA PARSING (MATLAB):
takes ~5 mins
(1) in eyes/data_processing/data/SUBJ/, open ASC file in vi and find line number of first data line (start_line)
(2) in MATLAB, directory eyes/data_processing/scripts/, run parsing script

	example call: ET_parse('SUBJ',start_line)

TASK CREATION:
(1) Find raw frames that encompass the period of time we're interested in processing.
(2) Move raw frames of interest to folder named for the activity
	eg frames of interest are numbered 1300-5300 in the inside walk for bwsdrdp6
	   so move frames 1300-5300 from raw_data/bwsdrdp6_images/task to
	   raw_data/bwsdrdp6_images/task_walk_1 for BOTH CAMERAS

IMAGE RECTIFICATION:
takes ~5 mins
(1) in cameras/image_rectification/scripts/, open DUMMY_rectifyframes_EXPTYPE.py and resave with subject name and experiment type named for the activity.

	example: subj='bwsdrdp', inputs=['task_walk_1']

(2) edit variables for current experiment and run image rectification script

	example call: python SUBJ_rectifyframes_drdp_targets.py


CAMERA TO EYE REGISTRATION:
basically instantaneous (watch for high rms error, we usually see ~5 pixels)
(1) in cameras/camera_registration/scripts/, run register camera script to estimate transformation between eyes and cameras using circle grids

        example call: python register_camera.py '../../raw_data/SUBJ_images' '../../stereo_calibration/data/SUBJ/' 1


EYE DATA PROCESSING (MATLAB):
takes ~10 mins
(1) in MATLAB, in eyes/data_processing/scripts/, run process script to determine cyclopean coordinate fixations 

	example call: ET_process('SUBJ')
******** these instructions are unclear
(2) maybe also run ET_process_no_epipolar for control
(3) run plotting function to check accuracy of processed results?

	example call: ET_plot_drdp('SUBJ_proc')


EYE DATA SYNCING (MATLAB):
takes ~1 min
(1) in eyes/data_processing/scripts/, run ET_frame_sync_EXPTYPE script to grab gaze coordinates time synced to each camera frame capture

	example call: ET_frame_sync_drdp('SUBJ')
		      ET_frame_sync_task('SUBJ')

E PICKING:
manual! takes as long as it takes (maybe 30 mins for task)
in cameras/e_picking/scripts/, run EPicker and click E's for all target directories
this step is not necessary to process the task disparities, but is necessary for the
target disparities!

DISPARITY ESTIMATION:
takes ~8 hours
(1) in cameras/disparity_estimation/scripts/, open DUMMY_disparitysgbm_EXPTYPE.py and resave with subject name and experiment type
(2) edit variables for current experiment
(3) to select frame numbers for disparity estimation, either 
	(a) paste in input lines (frame numbers) from MATLAB ET_frame_sync output if this is drdp_targets, or 
	(b) manually select relevant frames from task

	example call: python SUBJ_disparitysgbm_drdp_targets.py


EYE IMAGE AND DISPARITY RECONSTRUCTION:
takes ~4 hours
(1) in cameras/3d_reprojection/scripts/, open DUMMY_camera_to_eye_EXPTYPE.py and resave with subject name and experiment type
(2) edit variables for current experiment

	example call: python SUBJ_camera_to_eye_EXPTYPE.py

DISPARITY STATISTICS:


TODO: 
-parse missing frames into saccades, blinks, missing CRs (eccentric gaze and squinting). Look to see if more missing for outside than inside activities (IR reflections from sun)
-ET_process store un-epipolared data and look at differences

-remove vergence and recalc disparities

-medians at different eccentricities, also histograms

-vertical disparities
-better variance estimate than std?
-distributions of fixations in visual field and in depth

WALKING CATEGORIES

INSIDE: inside Evans
CAMPUS: majority man-made, some trees and grass
PARK: mixture of natural and man made, paved ground, lots of trees
NATURE: majority natural, unpaved ground, very minimal man made structures

for NAT task

CAMPUS1 = leaving Evans, walking to park area
NATURE1 = natural area near faculty club
PARK1 = walking by faculty club, to crossing bridge over creek
CAMPUS2 = walking to the campanile
PARK2 = walking in the trees by campanile