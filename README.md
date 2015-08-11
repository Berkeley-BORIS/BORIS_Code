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

(2) **CAMERA TO EYE REGISTRAION**: Clone this respository into session_data/raw/cam2eye_registration
	https://github.com/Berkeley-BORIS/BORIS_Cam2EyeRegData
	
Adding other raw data not on github:

(1) **STEREO CALIBRATION**: place stereo calibration data (~5GB) directory in session_data/raw/stereocalibration/

(2) **GAZE**: place eyelink ASC (SUBJ_RECORDING.asc) into session_data/raw/gaze/

(3) **SCENE IMAGES**: place stereo camera images in session_data/raw/scene

notes on structure:
===================

subj: a participant in the study

session: a single recording session

task: a subset of time within a session during which a particular task was performed

setting up Python environment:
===================

If you're using OSX, the code will work with a custom python environment

**Download Anaconda Python distribution:**

https://store.continuum.io/cshop/anaconda/

**Set up boris python environment:**

conda create -n boris --file=conda_requirements.txt: do from BORIS code directory to create boris anaconda environment

**activate it**

source activate boris

**add open cv**

conda install -c https://conda.binstar.org/menpo opencv

**add boris conda command line tool**

conda install -c https://conda.binstar.org/bwsprague boris

**set up environment for commance line interface**

boris config --root *path/to/session_data*

boris config --root /users/emily/BORIS/session_data

(this creates a .borisrc file in your home directory)

running standard analyses:
===================

**run set up for analysis**

python setup.py install

DATA ANALYSIS:

from code folder: 



