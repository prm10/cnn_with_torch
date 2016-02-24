require 'torch'
require 'gnuplot'
require 'optim'
local data_loader = require 'datasets/loader'

--transform data from the csv
-- local loader = data_loader.import_data()
-- loader:save_data()

local opt={
    batches=32,
    iterations=200,
    save_every=10,
    savefile='model_autosave',
    loadfile='model_autosave'
}
local loader = data_loader:load_data()
-- local input_batch,target_batch=loader:getBatchData(100)
-- print(#input_batch)
-- print(#target_batch)
-- gnuplot.plot(input_batch[{10,1,{},1}])
local model,criterion = require('models/create_model')()
local params, grad_params = model:getParameters()

function feval()
    ------------------ get minibatch -------------------
    local input,target = loader:getBatchData(opt.batches)
    --ff
    local output = model:forward(input)
    local loss = criterion:forward(model.output, target)
    model:zeroGradParameters()
    --bp
    criterion:backward(model.output, target)
    model:backward(input, criterion.gradInput)
    return loss, grad_params
end

--[
-- optimization stuff
local losses = {}
-- local optim_state = {learningRate = 1e-1}
local optim_state = {learningRate=0.01,momentum=0.9,weightDecay=1e-4}
local time = 0
for i = 1, opt.iterations do
    -- local _, loss = optim.adagrad(feval, params, optim_state)
    local timer = torch.Timer()
    local _, loss = optim.sgd(feval, params, optim_state)
    losses[#losses + 1] = loss[1]
    time = time + timer:time().real
    if i % opt.save_every == 0 then
        torch.save(opt.savefile, params)
        print(string.format("iteration %4d, loss = %6.8f,gradnorm = %6.4e, time = %6.4f", i, loss[1],grad_params:norm(),time))
        time=0
    end
end
--]]

params:copy(torch.load(opt.loadfile))
local last_batch=#loader.targets_mean
local len=500--loader.targets_mean[last_batch]:size(1)
local predict=torch.Tensor(len,loader.dim_target)
local target=torch.Tensor(len,loader.dim_target)
local x=torch.Tensor(1,1,loader.len_data,loader.dim_input)
for i=1,len do
  x:copy(loader.inputs[last_batch]:narrow(1,i,loader.len_data))
  local y1=loader.targets_mean[last_batch]:narrow(1,i,1)
  local y2=loader.targets_std[last_batch]:narrow(1,i,1)
  target[{{i},{}}]:copy(y1[{{},{17}}])
  -- target[{{i},{}}]:copy(
  --     torch.cat({
  --         y1[{{},{17}}],
  --         y1[{{},{20}}],
  --         y2[{{},{17}}],
  --         y2[{{},{20}}]
  --     },2)
  -- )
  predict[{i,{}}]:copy(model:forward(x))
  if i%torch.floor(len/10)==0 then
    print(i..'/'..len)
  end
end
local loss = criterion:forward(predict, target)
print(loss)
gnuplot.plot({predict[{{},1}]},{target[{{},1}]})
--]]
