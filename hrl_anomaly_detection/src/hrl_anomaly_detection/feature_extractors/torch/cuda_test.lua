require 'cutorch'

t1 = torch.CudaTensor(100):fill(0.5)
t2 = torch.CudaTensor(100):fill(1)
t1:add(t2)

require 'cunn'