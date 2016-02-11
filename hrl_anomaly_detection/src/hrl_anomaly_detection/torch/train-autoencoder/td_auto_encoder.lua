
local test  = require 'test'



--os.exit()

----------------------------------------------------------------------
-- trainable parameters
--

-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()

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
-- time-delay train model
--

print('==> training model')

local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local err = 0
local iter = 0


--print(params.maxiter,params.batchsize)

for t = 1,params.maxiter,params.batchsize do

   --------------------------------------------------------------------
   -- progress
   --
   iter = iter+1
   xlua.progress(iter, params.statinterval)

   --------------------------------------------------------------------
   -- create mini-batch
   --
   local inputs = torch.Tensor()
   local targets = torch.Tensor()
   local dataRange = torch.range(1, #rawData)
   for i = t,t+params.batchsize-1 do
      -- load new sample
      --print(i, t, rawData[i]:size() )
      --if rawData[i]==nil then
      --   print("jump")
      --   break
      --end

      local new_index = (i-1)%(#rawData)+1
      inputs = rawData[new_index]:clone()
      targets = rawData[new_index]:clone()
      --table.insert(inputs, input)
      --table.insert(targets, target)
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
         f = f + module:updateOutput(inputs[i], targets[i])

         -- gradients
         module:updateGradInput(inputs[i], targets[i])
         module:accGradParameters(inputs[i], targets[i])
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
      print('==> iteration = ' .. t .. ', average loss = ' .. err/params.statinterval)

      -- get training and test loss?
      local t = require 'temp'
      t = 5
      
      test()


      -- save/log current net
      local filename = paths.concat(params.dir, 'module.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      module1 = module:clone()
      netLighter(module1)
      torch.save(filename, module1)



      -- reset counters
      err = 0; iter = 0
   end
end



