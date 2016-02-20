require 'torch'
require "string_split"

local loader = {}
loader.__index=loader

function loader.load_data()
  local data = {}
  setmetatable(data, loader)
  data.inputs = {}
  data.targets_mean = {}
  data.targets_std = {}

  local function readcsv (filename)
    local data_return={}
    local f = torch.DiskFile(filename, "r")
    f:quiet()
    local line =  f:readString("*l")
    while line ~= '' do
      local data_line=string.split(line,",")
      table.insert(data_return,data_line)
      line = f:readString("*l")
    end
    return torch.Tensor(data_return)
  end

  for i=1,26 do
    data.inputs[i]=readcsv("../GL_data/cnn/data_"..i..".csv")
    data.targets_mean[i]=readcsv("../GL_data/cnn/targets_mean_"..i..".csv")
    data.targets_std[i]=readcsv("../GL_data/cnn/targets_std_"..i..".csv")
  end
  print('data load done.')
  collectgarbage()
  return data
end
