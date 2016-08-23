--*********************************************************************************
-- 2_model_multimodal.lua
-- two input multimodal model for personality traits analysis

-- Arulkumar (aruls@cse.iitm.ac.in)
-- team evolgen
--*********************************************************************************

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'io';
require 'lfs'
require 'cunn'
require 'cutorch'
dofile 'utilities.lua'
require 'nngraph';
require 'rnn'

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
    local MODELPATH = './01_07_2016_lstm_461_200_100/model#1.net--'
    if(path.isfile(MODELPATH)) then
      model = torch.load(MODELPATH);
      logger.info('Model loaded!')
      logger.info(model)
      io.read()
    else  
      -- audio features processing layer
      input1 = nn.Identity()()
      audio_view1 = nn.View(68 * 6)(input1)
      audio_linear1 = nn.Linear(68 * 6, 100)(audio_view1)
      audio_relu1 = nn.ReLU()(audio_linear1)
      
      -- video features processing layer
      input2 = nn.Identity()()
      video_conv1 = nn.VolumetricConvolution(3, 16, 3, 5, 5)(input2) -- 3x6x112x112 --> 16x4x108x108
      video_relu1 = nn.ReLU()(video_conv1)
      video_maxpool1 = nn.VolumetricMaxPooling(2, 2, 2)(video_relu1) -- 16x4x108x108 --> 16x2x54x54
      video_conv2 = nn.VolumetricConvolution(16, 16, 2, 5, 5)(video_maxpool1)  -- 16x2x54x54 --> 16x1x50x50
      video_relu2 = nn.ReLU()(video_conv2)
      video_maxpool2 = nn.VolumetricMaxPooling(1, 2, 2)(video_relu2) -- 16x1x50x50 --> 16x1x25x25
      video_conv3 = nn.VolumetricConvolution(16, 1, 1, 5, 5)(video_maxpool2)  -- 16x1x25x25 --> 1x1x21x21
      video_view = nn.View(21*21)(video_conv3)

      -- fusion layers
      join = nn.JoinTable(1)({audio_relu1, video_view})
      dropout1 = nn.Dropout(0.2)(join)
      linear3 = nn.Linear(541, 200)(dropout1)
      global_relu = nn.ReLU()(linear3)
      dropout2 = nn.Dropout(0.2)(global_relu)
      
      -- linear classifier
      linear4 = nn.Linear(200, 5)(dropout2)
      output = nn.Sigmoid()(linear4)
      model = nn.gModule({input1, input2}, {output})
            
      logger.info('Model created!')
      logger.info(model)
      
      graph.dot(model.fg, 'model', 'personalityTraitsModelTriamese')
    end
end
    
