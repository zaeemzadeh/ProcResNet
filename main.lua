--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'gnuplot'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge

local errLogger = optim.Logger('error.log')
errLogger:display(false)
trainTop1vec = torch.Tensor(opt.nEpochs,1):zero()
testTop1vec  = torch.Tensor(opt.nEpochs,1):zero()
--trainLossvec = torch.Tensor(opt.nEpochs,1):zero()

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)
   errLogger:add{['% train top1']    = trainTop1, ['% test top1']    = testTop1, ['% train loss']    = trainLoss}

    
   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', bestTop1, bestTop5)
   end
   print(' * Best result so far: ', bestTop1, bestTop5)
   -- plot logger
    trainTop1vec[epoch] = trainTop1
    testTop1vec[epoch]  = testTop1
    --trainLossvec[epoch] = trainLoss
    
    --gnuplot.figure(1)
    --gnuplot.plot({'Training Error', trainTop1vec:sub(1,epoch), '-'},
             --     {'Testing Error', testTop1vec:sub(1,epoch), '-'}) 
    --gnuplot.plot('Training Loss', trainLossvec:sub(1,epoch), '.')
    --gnuplot.axis({1,300,0,20})
    --gnuplot.epsfigure('error.eps')
    
   errLogger:style{['% train top1']    = '-', ['% test top1']    = '-'}
   errLogger:plot()

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
