require 'torch'
require 'gnuplot'
local data_loader = require 'datasets/loader'

-- local loader = data_loader.import_data()
-- loader:save_data()

local loader = data_loader:load_data()
-- for i=1,1000000 do
--   loader:getNextData()
-- end
local input_batch,target_batch=loader:getBatchData(10)
-- print(#input_batch)
-- print(#target_batch)
-- gnuplot.plot(input_batch[{10,1,{},1}])
