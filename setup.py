#------------------------------------------------------------

import subprocess
import logging
import zipfile
import sys, os

sys.path.append('./help_scripts')

import utils

#------------------------------------------------------------

trainDataDownloader = './help_scripts/train_val_getDataDirect.py'
testDataDownloader = './help_scripts/test_getDataDirect.py'

#----------------------------------
# -- prepare the logger
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('setup')
logger.info('downloading the training and validation data', "download data")

# download the training, validation, test data
#subprocess.call(['python', trainDataDownloader])
#subprocess.call(['python', testDataDownloader])

# extract the zip files of train, validation, test to appropriate directories
folders = ['train', 'validation', 'test']

for folder in folders:
	destfolder = os.path.join("./data", folder)
	passwd = ''
	if folder == 'test':
		passwd = '.chalearnLAPFirstImpressionsFirstRoundECCVWorkshop2016.'
		
	allfiles = os.listdir(os.path.join("./data", folder + 'zip'))
	filenumber = 1
	
	for file in allfiles:
		utils.progress(filenumber, allfiles.__len__());
		filenumber = filenumber + 1
		
		if file.endswith(".zip"):
			currentzipfile = os.path.join("./data", folder + 'zip', file)
			filename = os.path.splitext(os.path.basename(currentzipfile))[0]
			utils.mkdirs(os.path.join(destfolder, filename))
			with zipfile.ZipFile(currentzipfile,"r") as zip_ref:
				zip_ref.extractall(os.path.join(destfolder, filename), pwd=passwd)
