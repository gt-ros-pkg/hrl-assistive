require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'
require 'math'

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
--local loss = t.loss

----------------------------------------------------------------------
-- Save light network tools:
function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor() end
   module.gradWeight = nil
   module.output     = torch.Tensor()
   if module.fgradInput then module.fgradInput = torch.Tensor() end
   module.gradInput  = nil
end

function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end


----------------------------------------------------------------------
-- trainable parameters
--
-- get all parameters
x,dl_dx,ddl_ddx = model:getParameters()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local inputs = torch.Tensor( params.batchsize, params.nDim*params.timewindow)
local targets = torch.Tensor( params.batchsize, params.nDim*params.timewindow)

if params.cuda == true then
   inputs = inputs:cuda()
   targets = targets:cuda()
end

----------------------------------------------------------------------
-- time-delay train model
--
print(sys.COLORS.red ..  '==> defining training procedure')
local err = 0

local function train(iter, trainData)

   --------------------------------------------------------------------
   -- progress
   --
   --print(iter, params.statinterval)
   for t=1, trainData:size()[1], params.batchsize do
      xlua.progress(t, trainData:size()[1])

      --------------------------------------------------------------------
      -- create mini-batch
      --
      local idx = 1
      for i = t,t+params.batchsize-1 do
         inputs[idx]  = trainData[t]
         targets[idx] = trainData[t]
         idx = idx + 1
      end

      --------------------------------------------------------------------
      -- define eval closure
      --
      local feval = function()
         -- reset gradient/f
         local f = 0
         dl_dx:zero()

         -- estimate f and gradients, for minibatch
         for i = 1,inputs:size()[1] do

            -- f
            f = f + model:updateOutput(inputs[i], targets[i])

            -- gradients
            model:updateGradInput(inputs[i], targets[i])
            model:accGradParameters(inputs[i], targets[i])
         end

         -- normalize
         dl_dx:div(inputs:size()[1])
         f = f/inputs:size()[1]

         if f~=f then
            --print(f, #inputs)
            os.exit()
         end

         -- return f and df/dx
         return f,dl_dx
      end

   
      --------------------------------------------------------------------
      -- one SGD step with time delay
      --
      sgdconf = sgdconf or {learningRate = params.eta,
                            learningRateDecay = params.etadecay,
                            learningRates = etas,
                            momentum = params.momentum}
      _,fs = optim.sgd(feval, x, sgdconf)
      err = err + fs[1]
   end


   --------------------------------------------------------------------
   -- compute statistics / report error
   --
   if math.fmod(iter , params.statinterval) == 0 then

      -- report
      print('==> iteration = ' .. iter .. ', train average loss = ' .. err/params.statinterval)

      -- save/log current net
      local filename = paths.concat(params.dir, 'model.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      model1 = model:clone()
      netLighter(model1)
      torch.save(filename, model1)


      -- reset counters
      err = 0; 
   end

end

-- Export:
return train








