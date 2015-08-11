BORIS ( BinOcular Retinal Image Statistics ) Code
===================

code for analyzing the Berkeley dataset of visual movies

GETTING SET UP:

Download Anaconda Python distribution:
https://store.continuum.io/cshop/anaconda/


LINKING UP DATA:

Adding repositories:

(1) Clone the code repository (this one)
	https://github.com/Berkeley-BORIS/BORIS_Code

(2) Clone the camera to eye registration resposity into ../session_data/raw/cam2eye_registration
	https://github.com/Berkeley-BORIS/BORIS_Cam2EyeRegData
	
(3) Clone the gaze data repository into ../session_data/raw/gaze
	https://github.com/Berkeley-BORIS/BORIS_Cam2EyeRegData/TBD

Adding other raw datasets:

(1) place stereo calibration data directory in ../data/raw/stereocalibration/SUBJ/
(2) place eyelink ASC (SUBJ_RECORDING.asc) into ../data/raw/gaze/SUBJ/
(3) place stereo camera images in ../data/raw/scene/SUBJ/RECORDING
(4) in eyes/ipds create a text file called XXX.txt (subject's 3 initials) and enter ipd in cm

HOW TO INSTALL BORIS CONDA COMMANDLINE TOOL:

conda install bvs
check for updates somehow?

DATA ANALYSIS:

from code folder: python setup.py install



