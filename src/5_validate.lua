----------------------------------------------------------------------
-- 5_validate.lua
-- implements a validation procedure, to write predictions on 
-- validation data
--
-- Arulkumar (aruls@cse.iitm.ac.in)
-- team evolgen
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'cutorch'
require 'cunn'
dofile 'utilities.lua'
require 'io'
logger.outfile = opt.logFile
----------------------------------------------------------------------
logger.debug '==> defining test procedure'

-- validate function
function validate()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   
   --buffer for validation csv file
   validationPredictions = {}

   -- test over validation data
   logger.debug('==> validation on given set:')
   local totalError = 0  
   local totalValidationFiles = table.getn(validationaudiofiles)
   local totalTests = 10
   
  for i = 1, totalTests do
     
    for t = 1, totalValidationFiles do
      -- disp progress
      xlua.progress(t, totalValidationFiles)

      -- get new sample
      local mp4name = validationMp4names[t]      
      local inputaudio = validationAudioData[mp4name]

      if(not opt.LSTM) then
        target = validationTargets[mp4name]
      else
        target = torch.repeatTensor(validationTargets[mp4name] * opt.targetScaleFactor,VIDEOFEATURESROWS,1)
      end 
        
      if opt.type == 'double' then 
        inputaudio = inputaudio:double(); 
        target = target:double()
      elseif opt.type == 'cuda' then 
        inputaudio = inputaudio:cuda(); 
        target = target:cuda() 
      end
        
      --get the vide frames data
      local validationVideoData = prepareVideoFramesData({[mp4name]=1}, VIDEOFEATURESROWS, 'validation',opt.LSTM)        
      if(opt.type == 'cuda') then validationVideoData[mp4name] = validationVideoData[mp4name]:cuda() end
        
      -- estimate f
      local currentInput = {inputaudio, validationVideoData[mp4name]}
       
      -- test sample
      local pred = model:forward(currentInput)
      --do return end

      if(validationPredictions[mp4name] == nil) then validationPredictions[mp4name] = torch.Tensor(5):fill(0) end
      
      if(not opt.LSTM) then
        validationPredictions[mp4name] = validationPredictions[mp4name] + torch.squeeze( pred:double() / opt.targetScaleFactor )
      else
        validationPredictions[mp4name] = validationPredictions[mp4name] + torch.squeeze( (torch.mean(pred:double(),1) / opt.targetScaleFactor) )
      end    
    end
  end
   
  -- get the average of predictions from all the tests
  for mp4name, prediction in pairs(validationPredictions) do
    validationPredictions[mp4name] = prediction / totalTests
  end
   
  writeIntoCsvFile("valpred_model4/validation_predictions#" .. epoch .. ".csv", validationMp4names, validationPredictions)

  -- timing
  time = sys.clock() - time
  time = time / totalValidationFiles
  logger.debug("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

end
