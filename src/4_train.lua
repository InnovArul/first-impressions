----------------------------------------------------------------------
-- 4_train.lua
-- defines a training procedure
--
--   + it constructs mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to optmization method: SGD
--
-- Arulkumar (aruls@cse.iitm.ac.in)
-- team evolgen
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'pl'
require 'io'
require 'cutorch'
require 'cunn'
logger = dofile 'log.lua'
dofile 'utilities.lua'
logger.outfile = opt.logFile

----------------------------------------------------------------------
logger.debug '==> defining some tools'

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

----------------------------------------------------------------------
logger.debug '==> configuring optimizer'

--[[
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay
  --momentum = opt.momentum,
  --learningRateDecay = 1e-7
}
optimMethod = optim.adam
--]]

optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay
}
optimMethod = optim.sgd

----------------------------------------------------------------------
logger.debug '==> defining training procedure'

function train()

  ----------------------------------------------------------------------
  -- CUDA?
  if opt.type == 'cuda' then
    model:cuda()
    criterion:cuda()
  end

  -- Retrieve parameters and gradients:
  -- this extracts and flattens all the trainable parameters of the mode
  -- into a 1-dim vector
  parameters,gradParameters = model:getParameters()
  logger.info('total parameters : ' .. parameters:size(1))

  -- epoch tracker
  epoch = epoch or 1

  --[[
  -- learning rate decay code
  if epoch % opt.lrDecayEvery == 0 then
    local oldLearningRate = optimState.learningRate
    optimState = {learningRate = oldLearningRate * opt.lrDecayFactor, weightDecay = opt.weightDecay}
  end
  --]]

  -- local vars
  local time = sys.clock()
 
  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training()

  -- shuffle at each epoch
  local totalTrainFiles = table.getn(trainaudiofiles)
  shuffle = torch.randperm(totalTrainFiles)

  -- do one epoch
  logger.debug('==> doing epoch on training data:')
  logger.debug("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local totalError = 0;
  local trainPredictions = {}
  local trainFeatures = {}

  -- train batch wise
  for t = 1,totalTrainFiles,opt.batchSize do
    -- disp progress
    xlua.progress(t, totalTrainFiles)

    -- create mini batch
    local inputs = {}
    local targets = {}
    local mp4names = {}

    -- for each batch, collect the features
    for i = t, math.min(t+opt.batchSize-1,totalTrainFiles) do
      -- load new sample
      local inputfile = trainaudiofiles[shuffle[i]]
      local mp4name = path.basename(inputfile):gsub('.wav.csv', '')
      local inputaudio = trainAudioData[mp4name]

      if(not opt.LSTM) then
        local target = trainTargets[mp4name]
      else
        local target = torch.repeatTensor(trainTargets[mp4name] * opt.targetScaleFactor,VIDEOFEATURESROWS,1)
      end

      if opt.type == 'double' then 
        inputaudio = inputaudio:double(); 
        target = target:double()
      elseif opt.type == 'cuda' then 
        inputaudio = inputaudio:cuda();
        target = target:cuda() 
      end

      table.insert(inputs, inputaudio)
      table.insert(targets, target)
      table.insert(mp4names, mp4name)
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients  
      gradParameters:zero()

      -- f is the average of all criterions
      local f = 0

      -- evaluate function for complete mini batch
      for i = 1,#targets do
        --get the vide frames data
        local trainVideoData = prepareVideoFramesData({[mp4names[i]]=1}, VIDEOFEATURESROWS, 'train',opt.LSTM)
        if(opt.type == 'cuda') then trainVideoData[mp4names[i]] = trainVideoData[mp4names[i]]:cuda() end
            
        -- estimate f
        local currentInput = {inputs[i], trainVideoData[mp4names[i]]}
        local output = model:forward(currentInput)
        
        local err = criterion:forward(output, targets[i])
        f = f + err

        if(not opt.LSTM) then
            trainPredictions[mp4names[i]] = torch.squeeze( output:double() / opt.targetScaleFactor)
        else
          trainPredictions[mp4names[i]] = torch.squeeze( torch.mean(output:double(),1) / opt.targetScaleFactor)
        end
        -- estimate df/dW
        local df_do = criterion:backward(output, targets[i])
        model:backward(currentInput, df_do)
      end

      -- normalize gradients and f(X)
      totalError = totalError + f
      gradParameters:div(#targets)
      f = f/#targets

      -- return f and df/dX
      return f,gradParameters
    end

    -- optimize on current mini-batch
    if optimMethod == optim.asgd then
      _,_,average = optimMethod(feval, parameters, optimState)
    elseif optimMethod == optim.adam then
      _,loss = optimMethod(feval, parameters, optimState)
    else
      _,_,average = optimMethod(feval, parameters, optimState)
    end
  end

  writeIntoCsvFile(paths.concat(opt.save, "train_predictions.csv"), trainMp4names, trainPredictions)
   
  -- time taken
  time = sys.clock() - time
  time = time / totalTrainFiles
  logger.debug("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
  logger.info('train total error: ' .. totalError);

  -- save/log current net
  local filename = paths.concat(opt.save, 'model#' .. epoch .. '.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  logger.debug('==> saving model to '..filename)
  torch.save(filename, model)

  -- next epoch
  epoch = epoch + 1

end