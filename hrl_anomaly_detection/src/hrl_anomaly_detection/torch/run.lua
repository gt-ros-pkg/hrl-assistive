require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers

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
cmd:option('-threads', 4, 'threads')
cmd:option('-cuda', false, 'Enable cuda. Default:false')
cmd:option('-plot', false, 'Enable plot')

-- for all models:
cmd:option('-model', 'one', 'auto-encoder class: one | two | three | four')
--cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-midoutputsize', 20, 'size of the first hidden unit')
cmd:option('-midoutput2size', 10, 'size of the second hidden unit')
cmd:option('-outputsize', 5, 'size of hidden unit')
cmd:option('-lambda', 0.1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 1e-4, 'learning rate')
cmd:option('-batchsize', 64, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0, 'gradient momentum')
cmd:option('-maxiter', 1000000, 'max number of updates')
cmd:option('-timewindow', 4, 'size of time window')

-- logging:
cmd:option('-datafile', 'http://torch7.s3-website-us-east-1.amazonaws.com/data/tr-berkeley-N5K-M56x56-lcn.ascii', 'Dataset URL')
cmd:option('-statinterval', 10, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', false, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

params = cmd:parse(arg)

rundir = cmd:string('psd', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end

torch.manualSeed(params.seed)
torch.setnumthreads(params.threads)
--torch.setnumthreads(1024)



----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load cuda pkg')
if params.cuda == true then
   require 'cutorch'
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(1)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
   print(  cutorch.getDeviceProperties(cutorch.getDevice()) )
end




----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')
local data  = require 'data'
local train = require 'train'
local test  = require 'test'


----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

for t = 1,params.maxiter do

   train(t, data.trainData)
   test(t, data.testData)

end



