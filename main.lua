require 'torch'
local data_loader = require 'datasets/loader'

-- local loader = data_loader.import_data()
-- local loader:save_data()

local loader = data_loader:load_data()
for i=1,1000000 do
  loader:getNextData()
end
