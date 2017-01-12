import scipy.io.wavfile as wav
import subprocess
import os

# collect all file names from train folder
from os import listdir
from os.path import isfile, join
import exceptions
import numpy as np
import logging
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

basepath = ''
currdir = os.path.dirname(os.path.abspath(__file__))

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

##
# extractWavFile - to extract the wav audio file from mp4 files using ffmpeg
#
def extractWavFile(filepath):
    if(isfile(filepath)):
        if(filepath.lower().endswith('.mp4')):
            path, filename = os.path.split(filepath)
            print(basepath)
            destpath =  os.path.abspath(basepath) + 'audio/'
            print(destpath)
            mkdir_p(destpath)
            command = "ffmpeg -i " + filepath + " -ab 160k -ac 2 -ar 44100 -vn " + join(destpath, filename) + ".wav"
            subprocess.call(command, shell = True)
    else:
        allfiles = [f for f in listdir(filepath) if (f != '.' and f != '..')]
        for anyfile in allfiles:
            print join(filepath, anyfile)
            extractWavFile(join(filepath, anyfile))


##
# extractMFCCFeatures - to extract audio features from .wav files using pyAudioAnalysis toolkit
#         thanks to : https://github.com/tyiannak/pyAudioAnalysis
#         the featrures extracted are, as in https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
#
def extractMFCCFeatures(filepath):
    global sizeArray
    if(isfile(filepath)):
        if(filepath.lower().endswith('.wav')):
            path, filename = os.path.split(filepath)
            destpath =  os.path.abspath(basepath) + 'feat/'
            mkdir_p(destpath)
            (rate, sig) = audioBasicIO.readAudioFile(filepath);
            command = "python " + currdir + "/pyAudioAnalysis/audioAnalysis.py featureExtractionFile -i " + filepath + " -mw " + str(sig.shape[0]/float(rate)/5.5) + " -ms " +  str(sig.shape[0]/float(rate)/5.5) + " -sw 0.050 -ss 0.050 -o " + join(destpath, filename)
            print(command)
            subprocess.call(command, shell = True)
    else:
        allfiles = [f for f in listdir(filepath) if (f != '.' and f != '..')]
        for anyfile in allfiles:
            extractMFCCFeatures(join(filepath, anyfile))


def audioPreprocess(rootpath):
	global basepath
	basepath = rootpath            
	extractWavFile(basepath)
	basepath = rootpath + 'audio'
	sizeArray = None
	logging.basicConfig(filename=basepath+'_modifiedrate.log',level=logging.INFO)
	extractMFCCFeatures(basepath)

	# remove the .npy, .wav_st.csv files
	subprocess.call("rm -f " + rootpath + "audiofeat/*/*.npy", shell=True)
	subprocess.call("rm -f " + rootpath + "audiofeat/*/*.wav_st.csv", shell=True)
