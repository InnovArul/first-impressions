----------------------------------------------------------------------
-- 5_validate.lua
-- implements utility functions for loading data
--
-- Arulkumar (aruls@cse.iitm.ac.in)
-- team evolgen
----------------------------------------------------------------------

require 'torch'
require 'io'
require 'image'
require 'lfs'
local logger = dofile 'log.lua'
require 'pl'
require 'nn' -- for nn.Unsqueeze


--[[
   
   name: makeDataParallel
   @param
   @return creates Parallel models incase if we are using multiple GPUs
   
]]--
function makeDataParallel(model, nGPU)
    -- if the number of GOUs used is more than 1,
    -- create DataParallelTable
    if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1) --, true, true
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
      
      -- incase, if we are using DataParallelTable, create threads to avoid delay in kernel launching
      model:threads(function(idx)
          require 'nn'
          require 'cunn'
          require 'nngraph'
          require 'cutorch'
          
          cutorch.setDevice(idx)
      end);         
    end
    
    cutorch.setDevice(opt.GPU)

    return model
end


--[[
   
   name: sliceRange
   @param
   @return calculate the range of indices needed based on index and number of total splits
   
]]--
function sliceRange(nElem, idx, splits)
    -- calculate the count of common elements for all the GPUs
   local commonEltsPerMod = math.floor(nElem / splits)
   
    -- calculate the count of reamining elements for which the element-count shall be commonEltsPerMod + 1
   local remainingElts = nElem - (commonEltsPerMod * splits)
   
   -- according to current idx, how much "commonEltsPerMod + 1" elements are there?
   local commonPlusOneEltsCount = math.min(idx - 1, remainingElts)
   -- according to current idx, how much "commonEltsPerMod" elements are there?
   local commonEltsCount = (idx - 1) - commonPlusOneEltsCount 
   
   -- determine the start index
   local rangeStart = (commonPlusOneEltsCount * (commonEltsPerMod + 1)) + 
                        (commonEltsCount * commonEltsPerMod) + 1
                        
    -- determine the total elements for current index
   local currentElts = commonEltsPerMod
   if(idx <= remainingElts) then currentElts = commonEltsPerMod + 1 end

    -- return start index and elements count
   return rangeStart, currentElts
end


--[[
   UNUSED AS OF NOW
   name: cleanDPT
   @param
   @return clear the DataParallelTable and return new DataParallelTable
   
]]--
local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end

--[[
   
   name: saveDataParallel
   @param
   @return save the model with given filename
   
]]--
function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      -- save the model in first GPU
      temp_model = model:get(1):clearState()
      torch.save(filename, temp_model)
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else -- incase the given model is a plain model (due to nGPUs = 1)
      temp_model = model:clearState()
      torch.save(filename, temp_model)
   end
end

--[[
   UNUSED AS OF NOW
   name: loadDataParallel 
   @param
   @return load the saved DataParallelTable model
   
]]--
function loadDataParallel(filename, nGPU)
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

--[[
   
   name: localizeMemory
   @param
   @return copies the given tensor to GPU, incase GPU usage is forced
   
]]--
function localizeMemory(tensor)
  if(opt.useCuda) then
     newTensor = tensor:cuda();
  else
    newTensor = tensor;
  end
  
  return newTensor;
end

--[[

   name: getArrangedInputsForNGPUs 
   @param
   @return split the data into batches based on number of GPUs used
]]--
function getArrangedInputsForNGPUs(data, target, nGPUs, beginIndex, endIndex, randomOrder)
    local batches = {}
    local totalCount = 0
    if beginIndex and endIndex then 
      totalCount = endIndex - beginIndex + 1
    else
      totalCount = table.getn(target)
      beginIndex = 1
      endIndex = totalCount
    end
    
    local totalBatches = math.floor(totalCount / nGPUs)
    if(totalCount % nGPUs ~= 0) then totalBatches = totalBatches + 1 end
    --logger.trace('total number of inputs: ' .. totalCount .. ', batches : ' .. totalBatches)
    
    for index = 1, totalBatches do
        local startIndex, count = sliceRange(totalCount, index, totalBatches)

        local currentData = {}
        
        --remember that targets are always in default device index
        local currentTargets = torch.CudaTensor(count, 5)
        --xlua.progress(index, totalBatches)
        
        --get previous default dev index
        local prevDevId = cutorch.getDevice()
        for dataIndex = 1, count do
           currentDataIndex = randomOrder[(beginIndex - 1) + (startIndex- 1) + dataIndex]
           
           --copy to correct GPU
           local clonedData = {}
           cutorch.withDevice(dataIndex, function() 
                                            for i, internalData in ipairs(data[currentDataIndex]) do
                                               --copy the data to particular GPU
                                              if(torch.isTensor(internalData)) then
                                                --print (i .. ' is a tensor')
                                                clonedData[i] = internalData:clone()
                                              else
                                                local trainVideoData = prepareVideoFramesData({[internalData]=1}, VIDEOFEATURESROWS, 'train')
                                                clonedData[i] = trainVideoData[internalData]:cuda()
                                              end
                         
                                              --print ('data ' .. i .. ' : ' .. clonedData[i]:getDevice())                                              
                                            end
                                        end)
      
           --maybe, involve multi-threading?
           
           currentData[dataIndex] = clonedData
           
           currentTargets[dataIndex] = target[currentDataIndex]
           --io.read()
        end
        --set the default device index
        cutorch.setDevice(prevDevId);

        batches[index] = {
            input=currentData,
            labels=currentTargets
        }
    end

    return batches
end

--[[

   name: getInternalModel 
   @param
   @return get the internal model from wrapped model, if DataParallelTable is used
]]--
function getInternalModel(model)
    currentModel = model
    if torch.type(model) == 'nn.DataParallelTable' then
        currentModel = model:get(1)
    end
    return currentModel
end

--[[
-- readWavFeat = read the contents from wavfeat file into a tensor
--               append zeros if the tensor is having less than 3059 rows

NOT USED AS OF NOW!!!
CODE IS DIRECTLY ADDED IN loadFeaturesFromFolder
--]]
function readWavFeat(filePath)
  -- Read data from CSV to tensor
  ROWS = 3059
  COLS = 39
  local csvFile = io.open(filePath, 'r')  

  local data = torch.Tensor(1, ROWS, COLS):fill(0)

  local i = 0  
  for line in csvFile:lines('*l') do  
    i = i + 1
    local l = line:split(' ')
    for key, val in ipairs(l) do
      data[1][i][key] = val
    end
  end

  csvFile:close() 
  
  return data;
end


--[[
-- readTargetFile = read the contents from target file into a table
--]]
function readTargetFile(filePath)
  -- Read data from CSV to tensor
  local csvFile = io.open(filePath, 'r')  
  local header = csvFile:read() -- reads the first line, header file
  local targets = {}
  local mp4names = {}

  local i = 0  
  for line in csvFile:lines('*l') do  
    i = i + 1
    local l = line:split(',')
    local currentTarget = torch.Tensor(5)
    for key, val in ipairs(l) do
      if key ~= 1 then
        currentTarget[key-1] = val
      end
    end
    
    table.insert(mp4names, l[1])
    targets[l[1]] = currentTarget
  end

  csvFile:close() 
  
  return targets, mp4names;
end


--[[

--]]
function writeIntoCsvFile(filename, mp4names, predictions)

  csvfile = io.open(filename, 'w')
  csvfile:write("VideoName,ValueExtraversion,ValueAgreeableness,ValueConscientiousness,ValueNeurotisicm,ValueOpenness\n")
  --print(predictions)
  for index, mp4name in ipairs(mp4names) do
    local pred = predictions[mp4name]
    --print(pred)
    --io.read()
    csvfile:write(mp4name .. ',' .. 
                          string.format("%0.14f", pred[1]) .. ',' .. 
                          string.format("%0.14f", pred[2]) .. ',' .. 
                          string.format("%0.14f", pred[3]) .. ',' .. 
                          string.format("%0.14f", pred[4]) ..',' .. 
                          string.format("%0.14f", pred[5]) .. "\n")    
  end
  
  csvfile:close()
  
end

--[[
   
   name: table.map_length
   @param
   @return the number of keys in the table (i.e., effectively the length of the table)
   
]]--

function table.map_length(t)
    local c = 0
    for k,v in pairs(t) do
         c = c+1
    end
    return c
end


--[[
   
   name: table.getAllKeys
   @param
   @return the number of keys in the table (i.e., effectively the length of the table)
   
]]--

function table.getAllKeys(tbl)
    local keyset={}
    local n=0

    for k,v in pairs(tbl) do
      n=n+1
      keyset[n]=k
    end
    
    table.sort(keyset)
    return keyset;
end


--[[
   
   name: table.getValues
   @param
   @return get all the values for given keys
   
]]--

function table.getValues(tbl, keys)
    local values={}
    local n=0

    for index = 1, keys:size(1) do
      values[index]=tbl[keys[index]]
    end
    
    table.sort(values)
    return values;
end


--[[
   
   name: getRandomNumber
   @param
   @return a random number between lowe and upper, but without the number in exclude
   
]]--

function getRandomNumber(lower, upper, exclude)
    randNumber = math.random(lower, upper);
    while(randNumber == exclude) do
        randNumber = math.random(lower, upper);
    end
    return randNumber;
end


local Threads = require 'threads'
local t = Threads(10,
                  function()
                    require 'io'
                  end
                ) -- create a pool of 4 threads

--[[
loadAudioFeaturesFromFolder = recurse into subfolders and read the wavfeat files
--]]
function loadAudioFeaturesFromFolder(allfiles)
  local buffer = {}
  
  -- for all the files from the folder, call this function recursively
  for index, anyfile in ipairs(allfiles) do  
    -- if the given filepath corresponds to a file
    if(path.isfile(anyfile)) then
      -- if the filepath ends with .wavfeat
      if(anyfile:lower():match('.wav.csv')) then
        --extract actual mp4 file name
        logger.info('processing file ( ' .. index .. ' ) : ' .. anyfile)
        local filename = path.basename(anyfile)
        filename = filename:gsub(".wav.csv", "")
        local currentFilePath = anyfile
        local split = string.split
        local getAllKeys = table.getAllKeys
        local getValues = table.getValues
        
        t:addjob(
          function()
             -- Read data from CSV to tensor
              ROWS = 6
              COLS = 68
              local csvFile = io.open(currentFilePath, 'r')  

              --local data = torch.Tensor(1, ROWS, COLS)
              local data = torch.Tensor(ROWS, COLS)

              local i = 0  
              for line in csvFile:lines('*l') do  
                i = i + 1
                local l = split(line, ',')
                for key, val in ipairs(l) do
                  --if(key <= COLS + 1) then
                  --logger.debug(key .. ': ' .. val)
                  --data[1][i][key] = val
                  data[i][key] = val
                  --end
                end
              end

              csvFile:close() 
              
              return data;
          end,
          function(data)
            buffer[filename] = data
          end
        );
      end
    end
  end
  
  t:synchronize()
  
  return buffer
end


--[[
loadVideoFeaturesFromFolder = recurse into subfolders and read the wavfeat files
--]]
function loadVideoFeaturesFromFolder(allfiles, ROWS)
  local buffer = {}
  local lmBuffer = {}
  
  -- for all the files from the folder, call this function recursively
  for index, anyfile in ipairs(allfiles) do  
    -- if the given filepath corresponds to a file
    if(path.isfile(anyfile)) then
      -- if the filepath ends with .wavfeat
      if(anyfile:lower():match('_feat.txt')) then
        --extract actual mp4 file name
        logger.info('processing file ( ' .. index .. ' ) : ' .. anyfile)
        local filename = path.basename(anyfile)
        filename = filename:gsub("_feat.txt", ".mp4")
        local currentFilePath = anyfile
        local split = string.split
        local getAllKeys = table.getAllKeys
        local getValues = table.getValues
        
        --EYE GAZE
        local colIndices = torch.range(5, 10)
        
        --NON RIGID SHAPE
        colIndices = colIndices:cat(torch.range(363, 396))

        --ACTION UNITS
        colIndices = colIndices:cat(torch.range(397, 416))
        local COLS = colIndices:size(1)
        VIDEOFEATURESCOLS = COLS
        
        
        --ROUNDFACE LANDMARK POINTS
        --x
        local lmColIndices = torch.range(17, 43)
        --y
        lmColIndices = lmColIndices:cat(torch.range(85, 111))
        local LMCOLS = lmColIndices:size(1)
        VIDEOFEATURESLMCOLS = LMCOLS                
        
        t:addjob(
          function()
             -- Read data from CSV to tensor
              
              local csvFile = io.open(currentFilePath, 'r')  
              local header = csvFile:read() --read header
              
              local totalLines = 0
              for _ in csvFile:lines('*l') do
                totalLines = totalLines + 1
              end
              csvFile:close() 
              local data = torch.Tensor(1, totalLines, COLS):fill(0)
              local lmData = torch.Tensor(1, totalLines, LMCOLS):fill(0)
              
              csvFile = io.open(currentFilePath, 'r')  
              header = csvFile:read() --read header
              
              --validLineIndices = math.floor(totalLines / ROWS)
              --logger.debug('total lines = ' .. totalLines)
              --validLineCounter = 0
              local i = 0  
              
              for line in csvFile:lines('*l') do  
                i = i + 1

                local l = split(line, ',')
                local column = 1
                for index = 1,colIndices:size(1) do
                  key = colIndices[index]
                  data[1][i][column] = l[key]
                  column = column + 1
                end
                
                column = 1
                for index = 1,lmColIndices:size(1) do
                  key = lmColIndices[index]
                  lmData[1][i][column] = l[key]
                  column = column + 1
                end                
              end

              csvFile:close() 
              
              return data, lmData;
          end,
          function(data, lmData)
            buffer[filename] = data
            lmBuffer[filename] = lmData
          end
        );
      end
    end
  end
  
  t:synchronize()
  
  return buffer, lmBuffer
end

--[[

prepare the video data with random indices 

--]]
function prepareVideoData(rawData, ROWS)
  
  local allMp4Names = table.getAllKeys(rawData)
  local COLS = rawData[allMp4Names[1]--]:size(3)
  local preparedWholeData = {}
  
  for index, mp4name in ipairs(allMp4Names) do
    local currentData = rawData[mp4name]
    local totalLines = currentData:size(2)
    --logger.debug('file: ' .. mp4name .. ', lines: ' .. totalLines .. ', rows: ' .. ROWS)
    local validLineIndices = math.floor(totalLines / ROWS)
    
    local preparedData = torch.Tensor(1, ROWS, COLS)
    
    for row = 1, ROWS do
      local currentRowIndex = ((row-1) * validLineIndices) + torch.random(validLineIndices)
      preparedData[1][row] = currentData[1][currentRowIndex];
    end
    
    preparedWholeData[mp4name] = preparedData
  end
  
  return preparedWholeData
end

--[[

prepare the video frames data with random indices 

--]]
function prepareVideoFramesData(rawData, ROWS, category, isLSTM)
  
  local videoFramesPath = '/media/data/arul/personalitytraits/data/'.. category .. 'frames/'
  local allMp4Names = table.getAllKeys(rawData)
  local preparedWholeData = {}
  local unsqueezer = nn.Unsqueeze(2)
  
  for index, mp4name in ipairs(allMp4Names) do
    local currentFolderPath = videoFramesPath .. mp4name:gsub('.mp4', '')
    local allFiles = dir.getallfiles(currentFolderPath)
    local totalFiles = table.getn(allFiles)
    --logger.debug('folder: ' .. currentFolderPath .. ', files: ' .. totalFiles)
    local validFileIndices = math.floor(totalFiles / ROWS)
    
    local preparedData = nil
    
    for row = 1, ROWS do
      --frame index starts from 0
      local currentRowIndex = ((row-1) * validFileIndices) + torch.random(validFileIndices) - 1
      
      --framename determination
      framepath = path.join(currentFolderPath, string.format("frame_det_%06d.png", currentRowIndex))
      tempImage = image.load(framepath)
      
      if(preparedData == nil) then
        if(not isLSTM) then
          preparedData = torch.zeros(tempImage:size(1), ROWS, tempImage:size(2), tempImage:size(3))
        else
          preparedData = torch.zeros(ROWS, tempImage:size(1), tempImage:size(2), tempImage:size(3))
        end
      end
      
      tempImage = unsqueezer:forward(tempImage)
      if(not isLSTM) then
        preparedData[{{},{row},{},{}}] = tempImage
      else
        preparedData[{{row},{},{},{}}] = tempImage    
      end
    end
    
    preparedWholeData[mp4name] = preparedData
  end
  
  return preparedWholeData
end


--[[

prepare the video frames volume data with random indices 

--]]
function prepareVideoFramesVolume(rawData, ROWS, category)
  
  local videoFramesPath = '/media/data/arul/personalitytraits/data/'.. category .. 'frames/'
  local allMp4Names = table.getAllKeys(rawData)
  local preparedWholeData = {}
  local unsqueezer = nn.Unsqueeze(2)
  

  for index, mp4name in ipairs(allMp4Names) do
    local currentFolderPath = videoFramesPath .. mp4name:gsub('.mp4', '')
    local allFiles = dir.getallfiles(currentFolderPath)
    local totalFiles = table.getn(allFiles)
    --logger.debug('folder: ' .. currentFolderPath .. ', files: ' .. totalFiles)
    local validFileIndices = math.floor(totalFiles / ROWS)
    
    local preparedData = nil
    
    for row = 1, ROWS do
      --frame index starts from 0
      local currentRowIndex = ((row-1) * validFileIndices) + torch.random(validFileIndices - 4) - 1
      
      --framename determination
      local framepath = path.join(currentFolderPath, string.format("frame_det_%06d.png", currentRowIndex))
      local tempImage = image.load(framepath)
      
      if(preparedData == nil) then
        preparedData = torch.zeros(ROWS,tempImage:size(1),3,tempImage:size(2), tempImage:size(3))
      end
      
      tempImage = unsqueezer:forward(tempImage)
      
      tempImage = tempImage - meanImg
      
      preparedData[{{row},{},{1},{},{}}] = tempImage


      local framepath = path.join(currentFolderPath, string.format("frame_det_%06d.png", currentRowIndex+2))
      local tempImage = image.load(framepath)
      tempImage = unsqueezer:forward(tempImage)
      
      tempImage = tempImage - meanImg
      
      preparedData[{{row},{},{2},{},{}}] = tempImage
 
      local framepath = path.join(currentFolderPath, string.format("frame_det_%06d.png", currentRowIndex+4))
      local tempImage = image.load(framepath)
      tempImage = unsqueezer:forward(tempImage)
      
      tempImage = tempImage - meanImg
      
      preparedData[{{row},{},{3},{},{}}] = tempImage

    end
    
    preparedWholeData[mp4name] = preparedData
  end
  
  return preparedWholeData
end


--[[

prepare the video landmark data with random indices 

--]]
function prepareLandmarkData(rawData, ROWS)
  
  local allMp4Names = table.getAllKeys(rawData)
  local COLS = rawData[allMp4Names[1]]:size(3)
  local preparedWholeData = {}
  
  local SELECTABLEROWS = ROWS + 1
  for index, mp4name in ipairs(allMp4Names) do
    local currentData = rawData[mp4name]
    local totalLines = currentData:size(2)
    --logger.debug('file: ' .. mp4name .. ', lines: ' .. totalLines .. ', rows: ' .. ROWS)
    local validLineIndices = math.floor(totalLines / SELECTABLEROWS)
    
    local preparedData = torch.Tensor(1, SELECTABLEROWS, COLS):fill(0)
    
    for row = 1, ROWS + 1 do
      local currentRowIndex = ((row-1) * validLineIndices) + torch.random(validLineIndices)
      preparedData[1][row] = currentData[1][currentRowIndex];
    end
    
    preparedWholeData[mp4name] = preparedData
  end
  
  local newPreparedWholeData = {}
  --prepare the relative relocation vectors
  for mp4name, data in pairs(preparedWholeData) do
    local currentData = preparedWholeData[mp4name]
    local relocationVectors = torch.Tensor(1, ROWS, COLS):fill(0)
    
    for index = 2, SELECTABLEROWS do
      relocationVectors[1][index-1] = currentData[1][index] - currentData[1][index-1]
    end
    
    newPreparedWholeData[mp4name] = relocationVectors
  end
  
  return newPreparedWholeData
end