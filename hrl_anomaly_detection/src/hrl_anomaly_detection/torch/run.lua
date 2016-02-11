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
cmd:option('-threads', 2, 'threads')

-- for all models:
cmd:option('-model', 'linear', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
--cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-outputsize', 2, 'size of hidden unit')
cmd:option('-lambda', 0.1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 1, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0, 'gradient momentum')
cmd:option('-maxiter', 1000000, 'max number of updates')
cmd:option('-timewindow', 5, 'size of time window')

-- logging:
cmd:option('-datafile', 'http://torch7.s3-website-us-east-1.amazonaws.com/data/tr-berkeley-N5K-M56x56-lcn.ascii', 'Dataset URL')
cmd:option('-statinterval', 5000, 'interval for saving stats and models')
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



----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')
local data  = require 'data'
local train = require 'train'
local test  = require 'test'


----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

for t = 1,params.maxiter,params.batchsize do

   train(t, data.trainData)
   test(t, data.trainData)

end



