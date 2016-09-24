import scipy.io.wavfile as wav
import subprocess
import os

# collect all file names from train folder
from os import listdir
from os.path import isfile, join
import exceptions
import numpy as np
import logging

OpenfaceBase = "./OpenFace/build/bin"

##
# to create the parent directory (recursively if the fiolder structure is not existing already)
#
def mkdir_p(path):
    try: 
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise

# extract open face features
def extractOPENFACEFeatures(filepath):
    global sizeArray
    if(os.path.isfile(filepath)):
        if(filepath.lower().endswith('.mp4')):
            destbasepath = rootpath + 'frames'
            path, filename = os.path.split(filepath)
            destpath = destbasepath + '/' + filename.replace(".mp4","")
            mkdir_p(destpath)
            command = OpenfaceBase + '/FeatureExtraction' + ' -f ' + filepath + ' -rigid' + ' -simalign ' + destpath + ' -no2Dfp -no3Dfp -noMparams -noPose -noAUs -noGaze' + ' -q';
            print(command)
            subprocess.call(command, shell = True)
    else:
        allfiles = [f for f in os.listdir(filepath) if (f != '.' and f != '..')]
        for anyfile in allfiles:
            extractOPENFACEFeatures(filepath + '/' + anyfile)

def videoPreprocess(rootpath):
	basepath = rootpath
	extractOPENFACEFeatures(basepath)
