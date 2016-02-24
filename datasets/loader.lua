require "torch"
local dataset_path='../GL_data/cnn/'
local loader = {}
loader.__index=loader
loader.patches=23
loader.dim_input=25
loader.dim_target=1
loader.len_data=360*5;-- dataset cover over 5 hours
loader.len_target=6*20;-- target predict next 20 minutes
loader.index_time=1
loader.index_batch=1

function loader.import_data()
    local data = {}
    setmetatable(data, loader)
    data.inputs = {}
    data.targets_mean = {}
    data.targets_std = {}

    local function split (str, delimiter)
        if str==nil or str=='' or delimiter==nil then
            return nil
        end
        local result = {}
        for match in (str..delimiter):gmatch("(.-)"..delimiter) do
            table.insert(result, tonumber(match))
        end
        return result
    end

    local function readcsv (filename)
        local data_return={}
        local f = torch.DiskFile(filename, "r")
        f:quiet()
        local line =  f:readString("*l")
        while line ~= '' do
            local data_line=split(line,",")
            table.insert(data_return,data_line)
            line = f:readString("*l")
        end
        return torch.Tensor(data_return)
    end

    print("reading data")
    for i=1,loader.patches do
        print("data_"..i..".csv")
        data.inputs[i]=readcsv(dataset_path.."data_"..i..".csv")
        data.targets_mean[i]=readcsv(dataset_path.."target_mean_"..i..".csv")
        data.targets_std[i]=readcsv(dataset_path.."target_std_"..i..".csv")
        print('inputs: '..data.inputs[i]:size(1)..' '..data.inputs[i]:size(2))
        print('targets_mean: '..data.targets_mean[i]:size(1)..' '..data.targets_mean[i]:size(2))
        print('targets_std: '..data.targets_std[i]:size(1)..' '..data.targets_std[i]:size(2))
    end
    print('data import done.')
    collectgarbage()
    return data
end

function loader:save_data()
    torch.save(dataset_path..'for_torch/inputs.t7',self.inputs)
    torch.save(dataset_path..'for_torch/targets_mean.t7',self.targets_mean)
    torch.save(dataset_path..'for_torch/targets_std.t7',self.targets_std)
    print('data saved at: '..dataset_path..'for_torch/')
end

function loader:load_data()
    local data = {}
    setmetatable(data, loader)
    print('load data from: '..dataset_path..'for_torch/')
    data.inputs=torch.load(dataset_path..'for_torch/inputs.t7')
    data.targets_mean=torch.load(dataset_path..'for_torch/targets_mean.t7')
    data.targets_std=torch.load(dataset_path..'for_torch/targets_std.t7')
    local i1=0
    for i=1,#data.inputs do
      i1=i1+data.targets_std[i]:size(1)
    end
    print('get data with '..i1..' samples')
    return data
end

function loader:index_check()
    self.index_time=self.index_time+10
    if self.index_time>self.targets_mean[self.index_batch]:size(1) then
        -- print(self.index_batch..';'..self.index_time)
        self.index_batch=self.index_batch+1
        if self.index_batch>(#self.targets_mean-1) then
            self.index_batch=1
        end
        self.index_time=1
    end
end

function loader:getNextData()
  -- self.index_batch=1
  local x=self.inputs[self.index_batch]:narrow(1,self.index_time,self.len_data)
  local y1=self.targets_mean[self.index_batch]:narrow(1,self.index_time,1):div(2)
  local y2=self.targets_std[self.index_batch]:narrow(1,self.index_time,1)
  self:index_check()
  -- print(self.index_batch..','..self.index_time)
  return x,y1,y2
end

function loader:getBatchData(batches)
    local input_batch=torch.Tensor(batches,1,self.len_data,loader.dim_input)
    local target_batch=torch.Tensor(batches,loader.dim_target)
    for i=1,batches do
        local x,y1,y2=self:getNextData()
        input_batch[{{i},1,{},{}}]:copy(x)
        target_batch[{{i},{}}]:copy(y1[{{},17}])
        -- target_batch[{{i},{}}]:copy(
        --     torch.cat({
        --         y1[{{},{17}}],
        --         y1[{{},{20}}],
        --         y2[{{},{17}}],
        --         y2[{{},{20}}]
        --     },2)
        -- )
    end
    input_batch=torch.Tensor(input_batch)
    target_batch=torch.Tensor(target_batch)
    return input_batch,target_batch
end

-- function loader:getValData ()
--   local t=1
--   local len=self.targets_mean[#self.targets_mean]:size(1)
--   while  do
--     -- body...
--   end
-- end

return loader
