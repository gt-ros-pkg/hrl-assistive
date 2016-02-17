require 'torch'   -- torch
require 'hdf5'

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> downloading dataset')

----------------------------------------------------------------------
-- load data
--trainData = torch.load('./testh5py')
local myFile = hdf5.open('./test.h5py', 'r')
local trainData = myFile:read('trainingData'):all()
local testData = myFile:read('testData'):all()
myFile:close()

print(sys.COLORS.Green .. "Original training data size")
print( trainData:size() )
print(sys.COLORS.Green .. "Original test data size")
print( testData:size() )


----------------------------------------------------------------------
-- Time-delay extraction
params.nDim      = trainData:size(2)
params.nLength   = trainData:size(3)
params.inputsize = params.nDim*params.timewindow

local rawTrainData = torch.Tensor()
local rawTestData = torch.Tensor()

for i = 1,trainData:size(1) do
    for t = 1,trainData:size(3)-params.timewindow+1 do
        local singlesample = trainData:sub(i,i,1,params.nDim,t,t+params.timewindow-1):clone():reshape(1,params.nDim*params.timewindow)
        if singlesample==nil then
           print(singlesample)
        end

        if t==1 and i==1 then
           rawTrainData = singlesample
        else
           rawTrainData = torch.cat(rawTrainData, singlesample, 1)
        end
    end

    if i%100==0 then
       collectgarbage()
    end
end

for i = 1,testData:size(1) do
    for t = 1,testData:size(3),params.timewindow do
        local singlesample = testData:sub(i,i,1,params.nDim,t,t+params.timewindow-1):clone():reshape(1,params.nDim*params.timewindow)
        if singlesample==nil then
           print(singlesample)
        end

        if t==1 and i==1 then
           rawTestData = singlesample
        else
           rawTestData = torch.cat(rawTestData, singlesample, 1)
        end
    end

    if i%100==0 then
       collectgarbage()
    end
end


print(sys.COLORS.Green .. "Processed training data")
print(#rawTrainData)
print(sys.COLORS.Green .. "Processed test data")
print(#rawTestData)
print(sys.COLORS.Green .. "======================================")

-- Exports
return {
   trainData = rawTrainData,
   trainSingleDataLength = trainData:size(3)-params.timewindow+1,
   testData  = rawTestData,
   testSingleDataLength = torch.floor(testData:size(3)/params.timewindow)
   --mean = mean,
   --std = std,
   --classes = classes
}

