----------------------------------------------------------------------
-- doall.lua
-- 
-- An overall script to start the training and create the validation predictions
-- it creates the validation predictions once in 100 epochs
--
-- Arulkumar (aruls@cse.iitm.ac.in)
-- team evolgen
----------------------------------------------------------------------
require 'torch'
require 'lfs'
require 'io'
require 'cutorch'
logger = dofile 'log.lua'
VIDEOFEATURESROWS = 6
VIDEOFEATURESCOLS = 0
VIDEOFEATURESLMROWS = 6
VIDEOFEATURESLMCOLS = 0

opt = {}
opt.modelType = 'LSTMSpatial'
opt.save = paths.concat(lfs.currentdir(), opt.modelType .. '-' .. os.date("%d-%b-%Y-%H:%M:%S"))
opt.useCuda = true; --true / false
opt.type='cuda'
opt.optimization = 'SGD'   -- CG  | LBFGS  |  SGD   | ASGD
--opt.learningRate = 0.05
--opt.lrDecayFactor = 0.5
--opt.lrDecayEvery = 50
opt.learningRate = 0.05
opt.weightDecay = 5e-4
opt.momentum = 0.9
opt.learningRateDecay = 1e-4
opt.batchSize = 128
opt.forceNewModel = true
opt.logFile = paths.concat(opt.save, 'training.log')
opt.targetScaleFactor = 1
opt.nGPUs = 1
opt.GPU = 1
opt.LSTM = true
opt.useCuda = true

--define log file
logger.outfile = opt.logFile
paths.mkdir(opt.save)

-- include all the modules
dofile '1_data.lua'
dofile ('2_model_' .. opt.modelType .. '.lua')
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_validate.lua'

----------------------------------------------------------------------
logger.debug '==> training!'

-- log the options
logger.debug('options:')
for key, val in pairs(opt) do logger.debug(key, val) end
io.read()

-- create a model
create_model()

epoch = 1

--start training
while epoch <= 10000 do
   -- train for 100 epochs
   for index = 1, 100 do 
      train()
   end

   -- create validation predictions for the trained model
   validate()
end