BerkeleyVisionStats
===================

code for analyzing the Berkeley dataset of visual movies

GETTING SET UP:

Download Anaconda Python distribution:
https://store.continuum.io/cshop/anaconda/


LINKING UP DATA:

Adding other repositories:

(1) Clone the gaze data repository into ../data/raw/gaze
	https://github.com/eacooper/BerkeleyVisionStatsGazeData
(2) Clone the camera to eye registration resposity into ../data/raw/cam2eye_registration
	https://github.com/eacooper/BerkeleyVisionStatsCam2EyeReg


Adding other raw datasets:

(1) place stereo calibration data directory in ../data/raw/stereocalibration/SUBJ/
(2) place eyelink ASC (SUBJ_RECORDING.asc) into ../data/raw/gaze/SUBJ/
(3) place stereo camera images in ../data/raw/scene/SUBJ/RECORDING
(4) in eyes/ipds create a text file called XXX.txt (subject's 3 initials) and enter ipd in cm


