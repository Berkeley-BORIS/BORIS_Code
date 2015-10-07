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

**subj**: a participant in the study

kre, sah, tki

**session**: a single recording session

cafe, inside, nearwork, outside1, outside2

**task**: a subset of time within a session during which a particular task was performed

setting up Python environment:
===================

If you're using OSX, the code will work with a custom python environment

**Download Anaconda Python distribution:**

https://store.continuum.io/cshop/anaconda/

**add open cv -- currently using 2.4.10, not compatible with 3.0**

conda install -c https://conda.binstar.org/jjhelmus opencv

**add boris conda command line tool**

conda install -c https://conda.binstar.org/bwsprague boris

**Set up boris python environment:**

conda create -n boris --file=conda_requirements.txt: do from BORIS code directory to create boris anaconda environment

**activate it**

source activate boris

**if you want to use ipython notebook, add this too**

conda install ipython ipython-notebook ipython-qtconsole:

**set up environment for command line interface**

boris config --root *path/to/session_data*

boris config --root /users/emily/BORIS/session_data

(this creates a .borisrc file in your home directory)

DATA ANALYSIS
===================

**run set up for analysis (from code folder)**

python setup.py install

*if using python or ipython notebook rather than CLI, do:*

run setup.py install

import boris

running standard session-level pre-processing:
===================

**parsing the Eyelink files - gets gaze information from Eyelink file format**

boris parse *subj* *session*

boris parse kre cafe

or

boris parse_all

**calibrating stereo cameras - estimates intrinsics/extrinsics of the stereo cameras**

boris stereocalibrate *subj* *session*

boris stereocalibrate kre cafe

or

boris stereocalibrate_all

**registering camera to eyes - estimates the translation/rotation between the the left camera and the cyclopean eye (world origin)**

boris cam2eye *subj* *session*

or

boris cam2eye_all



