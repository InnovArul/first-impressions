----------------------------------------------------------------------
-- 1_data.lua
-- data loading (audio, video frames)

-- Vismay (vismayapatel@gmail.com)
-- team evolgen
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'io';
require 'lfs'
require 'cunn'
require 'cutorch'
dofile 'utilities.lua'
require 'nngraph';
require 'LSTM'

logger = require 'log'
logger.outfile = opt.logFile

--cudnn.benchmark = true;
--cudnn.fastest = true;
--cudnn.verbose = true;

--define fillcolors for different layers
COLOR_CONV = 'cyan';
COLOR_MAXPOOL = 'grey';
COLOR_RELU = 'lightblue';
COLOR_SOFTMAX = 'green';
COLOR_FC = 'orange';
COLOR_AUGMENTS = 'brown';

TEXTCOLOR = 'black';
NODESTYLE = 'filled';

-- hidden units, filter sizes (for ConvNet only):
local nstates = {64, 15, 5}
local filtrowsize = {91, 1}
local filtcolsize = {39, 5}
local rowstrides = {30, 1}
local colstrides = {1, 2}


function create_model()
 
      --[[
      -- code to load a model file
      local MODELPATH = './lstm/model#1.net--'
      if(path.isfile(MODELPATH)) then
      model = torch.load(MODELPATH);
      logger.info('Model loaded!')
      logger.info(model)
      io.read()
      else 
      end
      --]]

      audio_branch = nn.Sequential()
      audio_branch:add(nn.View(-1,68))
      audio_branch:add(nn.Linear(68,32)) -- 6x68 --> 6x32

      video_branch = nn.Sequential()
      video_branch:add(nn.SpatialConvolution(3,16,5,5)) -- 6x3x112x112 --> 6x16x108x108
      video_branch:add(nn.ReLU())
      video_branch:add(nn.SpatialMaxPooling(2,2)) -- 6x16x108x108 --> 6x16x54x54

      video_branch:add(nn.SpatialConvolution(16,16,7,7)) -- 6x16x54x54 --> 6x16x48x48
      video_branch:add(nn.ReLU())
      video_branch:add(nn.SpatialMaxPooling(2,2)) -- 6x16x48x48 --> 6x16x24x24

      video_branch:add(nn.SpatialConvolution(16,16,9,9)) -- 6x16x24x24 --> 6x16x16x16
      video_branch:add(nn.ReLU())
      video_branch:add(nn.SpatialMaxPooling(2,2)) -- 6x16x16x16 --> 6x16x8x8

      video_branch:add(nn.View(16*8*8))
      video_branch:add(nn.Linear(16*8*8,128)) -- 6x1024 --> 6x128
      video_branch:add(nn.ReLU())
      
      branches = nn.ParallelTable()
      branches:add(audio_branch)
      branches:add(video_branch)

      model = nn.Sequential()
      model:add(branches)
      model:add(nn.JoinTable(2))
      model:add(nn.Dropout(0.2))
      model:add(nn.View(-1,6,160))
      model:add(nn.LSTM(160,128))
      model:add(nn.View(-1,128))
      model:add(nn.Dropout(0.2))
      model:add(nn.Linear(128,5))
      model:add(nn.Sigmoid())

      logger.info('Model created!')
      logger.info(model)
    
end