require 'torch'   -- torch
require 'gnuplot'
require 'unsup'
require 'nn'

-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 2, 'threads')
cmd:option('-cuda', false, 'Enable cuda. Default:false')
cmd:option('-plot', true, 'Enable plot')

-- for all models:
cmd:option('-model', 'linear', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
--cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-outputsize', 35, 'size of hidden unit')
cmd:option('-timewindow', 4, 'size of time window')

-- logging:
cmd:option('-statinterval', 10, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', false, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

params = cmd:parse(arg)


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load data')
local data  = require 'data'

local testData = data.testData
local singleLength = data.testSingleDataLength

local nDim     = testData[1]:size()[1]/params.timewindow
local times    = torch.Tensor(singleLength)
local inputs   = torch.Tensor(singleLength,testData[1]:size()[1])
local preds    = torch.Tensor(singleLength,testData[1]:size()[1])
local features = torch.Tensor(singleLength,params.outputsize)

local viz_inputs = torch.Tensor(singleLength, nDim)
local viz_preds  = torch.Tensor(singleLength, nDim)

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Load model!')

-- save/log current net
local filename = paths.concat(params.dir, 'model.net')
print('==> loading model to '..filename)
local model = torch.load(filename)

for t = 1,testData:size(1),singleLength do

    -- get data
    local idx = 1
    for i=t, t+singleLength-1 do
        inputs[idx] = testData[i]
        idx         = idx + 1                
    end

    -- compute prediction and reduced feature
    for i=1, singleLength do 
        model.encoder:updateOutput(inputs[i])
        model.decoder:updateOutput(model.encoder.output)
        preds[i]    = model.decoder.output
        features[i] = model.encoder.output


        for j=1, nDim do
            viz_inputs[i][j] = inputs[i][j*params.timewindow]
            viz_preds[i][j]  = preds[i][j*params.timewindow]
        end
        times[i]  = i

    end
    os.exit()

    -- visualize original inputs & predictions
    local figure1 = gnuplot.figure(1)
    gnuplot.plot( {'inputs', times, viz_inputs:t()[1], '-'},
                  {'preds', times, viz_preds:t()[1], '-'})

    print(times:size(), features:size())

    -- visualize reduced features
    local figure2 = gnuplot.figure(2)
    gnuplot.plot(
        {'1', features:t()[1], '-'},
        {'2', features:t()[2], '-'},
        {'3', features:t()[3], '-'},
        {'4', features:t()[4], '-'},
        {'5', features:t()[5], '-'},
        {'6', features:t()[6], '-'},
        {'7', features:t()[7], '-'},
        {'8', features:t()[8], '-'},
        {'9', features:t()[9], '-'},
        {'10', features:t()[10], '-'})

    collectgarbage()
end
