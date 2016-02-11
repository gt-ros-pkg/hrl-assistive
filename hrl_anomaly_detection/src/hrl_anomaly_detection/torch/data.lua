require 'torch'   -- torch
require 'hdf5'

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> downloading dataset')

----------------------------------------------------------------------
-- load data
--trainData = torch.load('./testh5py')
local myFile = hdf5.open('/home/dpark/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/src/hrl_anomaly_detection/torch/test.h5py', 'r')
local dataset = myFile:read('data'):all()
myFile:close()

print( dataset:size() )



----------------------------------------------------------------------
-- Time-delay extraction
--local time_window = 5
local nDim = dataset:size(2)

params.inputsize  = nDim*params.timewindow
--params.batchsize  = dataset:size(3)-params.timewindow+1 

local rawData = {} -- torch.Tensor()

for i = 1,dataset:size(1) do
    local singleSamples = torch.Tensor()
    for t = 1,dataset:size(3),params.timewindow do
        local singlesample = dataset:sub(i,i,1,nDim,t,t+params.timewindow-1):clone():reshape(1,nDim*params.timewindow)
        if singlesample==nil then
           print(singlesample)
        end

        if t==1 then
           singleSamples = singlesample
        else
           singleSamples = torch.cat(singleSamples, singlesample, 1)
        end
    end

    table.insert(rawData, singleSamples)
end



-- Exports
return {
   trainData = rawData,
   --testData = testData,
   --mean = mean,
   --std = std,
   --classes = classes
}

