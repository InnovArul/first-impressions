----------------------------------------------------------------------
-- 3_loss.lua
-- define a mean squared error loss function
--
-- Arulkumar (aruls@cse.iitm.ac.in)
-- team evolgen
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions
logger = dofile 'log.lua'
logger.outfile = opt.logFile

-- 10-class problem
noutputs = 5

----------------------------------------------------------------------
logger.debug '==> define loss'

criterion = nn.MSECriterion()
criterion.sizeAverage = false
----------------------------------------------------------------------
logger.debug '==> here is the loss function:'
logger.debug(criterion)
