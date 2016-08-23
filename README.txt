ChaLearn LAP. Apparent Personality Analysis: First Impressions
------------------------------------------------------------------------
used OS: Ubuntu 14.04

Prerequisites for execution:
------------------------------------------------------------------------

To extract audio features:
--------------------------

1. install ffmpeg
https://git.ffmpeg.org/gitweb/ffmpeg.git
(or)
try "sudo apt-get install ffmpeg"

2. python 2.7 (<--tested) (or above) 

additional packages needed:
---------------------------

pip install --user eyed3
pip install --user mlpy
pip install --user "scikit-learn>=0.16,<0.17"  
--> (make sure that sklearn version is 0.16.1)

3. pyAudioAnalysis (already available in Dropbox folder)
Github: https://github.com/tyiannak/pyAudioAnalysis

4. OpenFace
Installation: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation

   - make sure that open face is installed in the folder "OpenFace-master" of same directory of test files
   i.e., open face should be built and the file "FeatureExtraction" should 
   be available in the folder "./OpenFace-master/build/bin"

5. Torch7
Installation: http://torch.ch/docs/getting-started.html

important modules of Torch7:
----------------------------
nn, cunn, cutorch, nngraph, rnn, image, optim, xlua, io
CUDA 7.0 or above

===========================================================================================================================

EXECUTION PROCEDURE:

Preprocessing steps:
---------------------

1. in commandline, go to "data" folder : 

2. make sure that the extracted train, validation videos are available in 
"data/train", "data/validation" folders respectively. These foldernames are hardcoded inside 
(some places of) training script. so, the folder names shall be strictly followed.

-- "data/train" folder should contain folders named "train80_01, train80_02 and so on"
-- "data/validation" folder should contain folders named "validation80_01, validation80_02 and so on"

3. PREPROCESSING FOR TRAIN DATA
---------------------------------
AUDIO:
------
   i. at line 14 of python preprocessing_audiofeats.py,
   rootpath = './train'

   ii. execute "python preprocessing_audiofeats.py" (* refer prerequisites for needed installations )
   
VIDEO:
------

   i. at line 14 of python preprocessing_videofeats.py,
   rootpath = './train'

   ii. execute "python preprocessing_videofeats.py" (* refer prerequisites for needed installations )
   
4. PREPROCESSING FOR VALIDATION DATA
-------------------------------------

AUDIO:
------
   i. at line 14 of python preprocessing_audiofeats.py,
   rootpath = './validation'

   ii. execute "python preprocessing_audiofeats.py" (* refer prerequisites for needed installations )
   
VIDEO:
------

   i. at line 14 of python preprocessing_videofeats.py,
   rootpath = './validation'

   ii. execute "python preprocessing_videofeats.py" (* refer prerequisites for needed installations )

5. After preprocessing, there will be 4 new folders created under "data" folder namely,

   "data/trainaudiofeat" - training audio features
   "data/validationaudiofeat" - validation audio features
   "data/trainframes" - training video features
   "data/validationframes" - validation video features
   
6. To train the model,
--------------------------

--> in commandline, go to "src" folder,
--> execute "th doall.lua"

************************************************************************
