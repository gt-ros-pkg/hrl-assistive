require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- model:
local t = require 'model'
local model = t.model

-- Batch test:
local inputs = torch.Tensor() -- get size from data
local targets = torch.Tensor()

local err = 0

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function test(t, testData)

   -- test over test data
   --print(sys.COLORS.red .. '==> testing on test set:')
   for k = 1,#testData,params.batchsize do
      -- disp progress
      --xlua.progress(k, #testData)

      -- create mini batch
      local f = 0
      for i = k,k+params.batchsize-1 do
          local new_index = (i-1)%(#testData)+1
          inputs = testData[new_index]:clone()
          targets = testData[new_index]:clone()
      end

      for i = 1,inputs:size()[1] do
          -- f
          f = f + model:updateOutput(inputs[i], targets[i])
      end

      -- normalize
      err = err + f/inputs:size()[1]

   end

   --------------------------------------------------------------------
   -- compute statistics / report error
   --
   if math.fmod(t , params.statinterval) == 0 then
      print('==> iteration = ' .. t .. ', test average loss = ' .. err/params.statinterval)
      err = 0
   end


end

-- Export:
return test


