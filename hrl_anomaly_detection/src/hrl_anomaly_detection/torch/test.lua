require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- model:
local t = require 'model'
local model = t.model

-- Batch test:
local inputs = torch.Tensor( params.batchsize, params.nDim*params.timewindow)
local targets = torch.Tensor( params.batchsize, params.nDim*params.timewindow)
if params.cuda == true then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end

local err = 0
test_losses = {}

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function test(iter, testData)

   -- test over test data
   --print(sys.COLORS.red .. '==> testing on test set:')
   local iter_t=0
   for t = 1,testData:size()[1],params.batchsize do

      -- disp progress
      iter_t = iter_t + 1
      --xlua.progress(t, testData:size()[1])

      -- batch fits?
      if (t + params.batchsize - 1) > testData:size()[1] then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+params.batchsize-1 do
         inputs[idx] = testData[i]
         targets[idx] = testData[i]
         idx = idx + 1
      end

      local f = 0
      for i = 1,params.batchsize do
          -- f
          f = f + model:updateOutput(inputs[i], targets[i])
      end

      -- normalize
      err = err + f/params.batchsize

   end

   --------------------------------------------------------------------
   -- compute statistics / report error
   --
   if math.fmod(iter , params.statinterval) == 0 then
      print('==> iteration = ' .. iter .. ', test average loss = ' .. err/params.statinterval/iter_t)

      if params.plot == true then
         table.insert(test_losses, err/params.statinterval/iter_t)

         local Y1 = torch.Tensor(train_losses)
         local Y2 = torch.Tensor(test_losses)

         if #test_losses > 3 then
            local iter_range = torch.range(1,#test_losses)                                
            local figure = gnuplot.figure(1)
            gnuplot.plot({'train',Y1,'-'},{'test', Y2, '-'})
            --gnuplot.plotflush()
         end
      end


      err = 0
   end


end

-- Export:
return test


