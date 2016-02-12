require 'torch'   -- torch
require 'hdf5'

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> downloading dataset')

----------------------------------------------------------------------
-- load data
--trainData = torch.load('./testh5py')
local myFile = hdf5.open('~/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/src/hrl_anomaly_detection/torch/test.h5py', 'r')
local trainData = myFile:read('trainingData'):all()
local testData = myFile:read('testData'):all()
myFile:close()

print( trainData:size() )
print( testData:size() )


----------------------------------------------------------------------
-- Time-delay extraction
--local time_window = 5
local nDim = trainData:size(2)

params.inputsize  = nDim*params.timewindow
--params.batchsize  = trainData:size(3)-params.timewindow+1 

local rawTrainData = {}
local rawTestData = {} 

for i = 1,trainData:size(1) do
    local singleSamples = torch.Tensor()
    for t = 1,trainData:size(3),params.timewindow do
        local singlesample = trainData:sub(i,i,1,nDim,t,t+params.timewindow-1):clone():reshape(1,nDim*params.timewindow)
        if singlesample==nil then
           print(singlesample)
        end

        if t==1 then
           singleSamples = singlesample
        else
           singleSamples = torch.cat(singleSamples, singlesample, 1)
        end
    end

    table.insert(rawTrainData, singleSamples)
end

for i = 1,testData:size(1) do
    local singleSamples = torch.Tensor()
    for t = 1,testData:size(3),params.timewindow do
        local singlesample = testData:sub(i,i,1,nDim,t,t+params.timewindow-1):clone():reshape(1,nDim*params.timewindow)
        if singlesample==nil then
           print(singlesample)
        end

        if t==1 then
           singleSamples = singlesample
        else
           singleSamples = torch.cat(singleSamples, singlesample, 1)
        end
    end

    table.insert(rawTestData, singleSamples)
end


-- Exports
return {
   trainData = rawTrainData,
   testData  = rawTestData,
   --mean = mean,
   --std = std,
   --classes = classes
}

