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
local inputs = torch.Tensor()
local targets = torch.Tensor()


----------------------------------------------------------------------
-- time-delay train model
--
print(sys.COLORS.red ..  '==> defining training procedure')

local iter = 0
local err = 0

local function train(t, trainData)

   --------------------------------------------------------------------
   -- progress
   --
   iter = iter+1
   xlua.progress(iter, params.statinterval)

   --------------------------------------------------------------------
   -- create mini-batch
   --
   for i = t,t+params.batchsize-1 do
      local new_index = (i-1)%(#trainData)+1
      inputs = trainData[new_index]:clone()
      targets = trainData[new_index]:clone()
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



   --------------------------------------------------------------------
   -- compute statistics / report error
   --
   if math.fmod(t , params.statinterval) == 0 then

      -- report
      print('==> iteration = ' .. t .. ', train average loss = ' .. err/params.statinterval)

      -- save/log current net
      local filename = paths.concat(params.dir, 'model.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      model1 = model:clone()
      netLighter(model1)
      torch.save(filename, model1)


      -- reset counters
      err = 0; iter = 0
   end

end

-- Export:
return train








