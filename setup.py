#------------------------------------------------------------

import subprocess
import logging
import zipfile
import sys, os

sys.path.append('./help_scripts')
sys.path.append('./data')

import utils
import preprocessing_audiofeats as audioprocess
import preprocessing_videofeats as videoprocess
import progressbar

#------------------------------------------------------------

trainDataDownloader = './help_scripts/train_val_getDataDirect.py'
testDataDownloader = './help_scripts/test_getDataDirect.py'
audioPreprocessor = './data/preprocessing_audiofeats.py'
videoreprocessor = './data/preprocessing_videofeats.py'

#----------------------------------
# -- prepare the logger
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'setup'))
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(0)
logger.setLevel(0)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(0)
logger.addHandler(consoleHandler)
logger.info('downloading the training and validation data')

# download the training, validation, test data
#subprocess.call(['python', trainDataDownloader])
#subprocess.call(['python', testDataDownloader])

# extract the zip files of train, validation, test to appropriate directories
folders = ['train', 'validation'] #, 'test']

def newProgressBar():
	bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', 
										progressbar.Bar(),
										' (', progressbar.ETA(), ') ',
							])
	return bar

for folder in folders:
	destfolder = os.path.join("./data", folder)
	passwd = ''
	if folder == 'test':
		passwd = '.chalearnLAPFirstImpressionsFirstRoundECCVWorkshop2016.'
		
	allfiles = os.listdir(os.path.join("./data", folder + 'zip'))
	bar = newProgressBar()
	
	logger.info('processing folder ' + folder)
	for file in bar(allfiles):
		if file.endswith(".zip"):
			currentzipfile = os.path.join("./data", folder + 'zip', file)
			filename = os.path.splitext(os.path.basename(currentzipfile))[0]
			utils.mkdirs(os.path.join(destfolder, filename))
			with zipfile.ZipFile(currentzipfile,"r") as zip_ref:
				zip_ref.extractall(os.path.join(destfolder, filename), pwd=passwd)
				
	audioprocess.audioPreprocess(destfolder)

#subprocess.call(['python', audioPreprocessor])
#subprocess.call(['python', videoreprocessor])
