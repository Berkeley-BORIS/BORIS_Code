BORIS ( BinOcular Retinal Image Statistics ) Code
===================

code for analyzing the Berkeley dataset of visual movies

GETTING SET UP
===================

linking up data:
===================

Adding repositories from Github:

(1) **CODE**: Clone this repository (this one)
	https://github.com/Berkeley-BORIS/BORIS_Code

(2) **CAMERA TO EYE REGISTRAION**: Clone this respository into ../session_data/raw/cam2eye_registration
	https://github.com/Berkeley-BORIS/BORIS_Cam2EyeRegData
	
Adding other raw data not on github:

(1) **STEREO CALIBRATION**: place stereo calibration data (~5GB) directory in ../session_data/raw/stereocalibration/

(2) **GAZE**: place eyelink ASC (SUBJ_RECORDING.asc) into ../session_data/raw/gaze/

(3) **SCENE IMAGES**: place stereo camera images in ../session_data/raw/scene

setting up Python environment:
===================

Download Anaconda Python distribution:
https://store.continuum.io/cshop/anaconda/

conda install boris
**check for updates somehow?**

DATA ANALYSIS:

from code folder: python setup.py install



