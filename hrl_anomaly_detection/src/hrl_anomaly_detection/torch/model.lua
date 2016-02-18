require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'unsup'

----------------------------------------------------------------------
-- create model
--

local encoder = nn.Sequential()
local decoder = nn.Sequential()

if params.model == 'one' then
   -- params
   local inputSize = params.inputsize 
   local outputSize = params.outputsize

   -- encoder
   encoder:add(nn.Linear(inputSize,outputSize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputSize))

   -- decoder
   decoder:add(nn.Linear(outputSize,inputSize))

elseif params.model == 'two' then

   -- params
   local inputSize    = params.inputsize 
   local midInputSize = params.midoutputsize
   local outputSize   = params.outputsize

   -- encoder
   encoder:add(nn.Linear(inputSize,midInputSize))
   encoder:add(nn.Linear(midInputSize,outputSize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputSize))

   -- decoder
   decoder:add(nn.Linear(outputSize,midInputSize))
   decoder:add(nn.Linear(midInputSize,inputSize))

elseif params.model == 'three' then

   -- params
   local inputSize    = params.inputsize 
   local midInputSize = params.midoutputsize
   local midInput2Size = params.midoutput2size
   local outputSize   = params.outputsize

   -- encoder
   encoder:add(nn.Linear(inputSize,midInputSize))
   encoder:add(nn.Linear(midInputSize,midInput2Size))
   encoder:add(nn.Linear(midInput2Size,outputSize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputSize))

   -- decoder
   decoder:add(nn.Linear(outputSize,midInput2Size))
   decoder:add(nn.Linear(midInput2Size,midInputSize))
   decoder:add(nn.Linear(midInputSize,inputSize))

end

-- complete model
local model = unsup.AutoEncoder(encoder, decoder, params.beta)

-- verbose
print('==> constructed single-layer auto-encoder')


if params.cuda == true then
   model.encoder:cuda()
   model.decoder:cuda()
   model.loss:cuda()
end


-- return package:
return {
   model = model,
   --loss = loss,
}


