require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'unsup'

----------------------------------------------------------------------
-- create model
--
-- params
local inputSize = params.inputsize 
local outputSize = params.outputsize

-- encoder
local encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,outputSize))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputSize))

-- decoder
local decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize,inputSize))

-- complete model
local model = unsup.AutoEncoder(encoder, decoder, params.beta)

-- verbose
print('==> constructed linear auto-encoder')


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


