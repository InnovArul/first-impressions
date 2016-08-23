----------------------------------------------------------------------
-- 1_data.lua
-- data loading (audio, video frames)

-- Arulkumar (aruls@cse.iitm.ac.in)
-- team evolgen
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
dofile 'utilities.lua'
logger = dofile 'log.lua'
logger.outfile = opt.logFile
require 'pl'

-- file paths
trainAudioFeaturePath = '../data/trainaudiofeat';
validationAudioFeaturePath = '../data/validationaudiofeat';

--collect all file/folder names in current directory 
trainaudiofiles = dir.getallfiles(trainAudioFeaturePath)
validationaudiofiles = dir.getallfiles(validationAudioFeaturePath)

logger.info('loading the targets')

-- read training file's target values
trainTargetFile = '../data/training_gt.csv'
trainTargets, trainMp4names = readTargetFile(trainTargetFile)
validationTargetFile = '../data/validation_gt.csv'
validationTargets, validationMp4names = readTargetFile(validationTargetFile)

-- buffer to hold data points
logger.info('loading the train audio data')
trainAudioData = loadAudioFeaturesFromFolder(trainaudiofiles)
logger.info('loading the validation audio data')
validationAudioData = loadAudioFeaturesFromFolder(validationaudiofiles)

--[[
-- CODE FOR LOADING VIDEO FEATURES
--trainVideoFeaturePath = '../data/trainvideofeat';
--validationVideoFeaturePath = '../data/validationvideofeat';
--trainvideofiles = dir.getallfiles(trainVideoFeaturePath)
--validationvideofiles = dir.getallfiles(validationVideoFeaturePath)
--logger.info('loading the train video data')
--trainRawVideoData, trainRawVideoLmData = loadVideoFeaturesFromFolder(trainvideofiles)
--logger.info('loading the validation video data')
--validationRawVideoData, validationRawVideoLmData = loadVideoFeaturesFromFolder(validationvideofiles)
--]]

--[[
-- mean image for traing mean subtraction
meanImg = torch.Tensor(3,112,112)
meanImg[1]:fill(0.49)
meanImg[2]:fill(0.45)
meanImg[3]:fill(0.42)
--]]