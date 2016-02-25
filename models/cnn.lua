require 'nn'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel()
  -- local function basicblock(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  --   return nn.Sequential()
  --     :add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
  --     :add(SBatchNorm(nOutputPlane))
  --     :add(ReLU(true))
  --     :add(Max(torch.floor(kW/2),torch.floor(kH/2),dW,dH,torch.floor(padW/2),torch.floor(padH/2)))
  -- end

-- x=torch.Tensor(20,1,1800,25)
local frame=torch.Tensor({2,5,10,20,10})
local model = nn.Sequential()
    :add(Convolution(1,frame[1],1,5,1,1,0,0))
    -- :add(SBatchNorm(frame[1]))
    :add(ReLU(true))
    :add(Max(1,10,1,10))

    :add(Convolution(frame[1],frame[2],1,5,1,1,0,0))
    -- :add(SBatchNorm(frame[2]))
    :add(ReLU(true))
    :add(Max(1,10,1,10))

    :add(Convolution(frame[2],frame[3],25,17,1,1,0,0))
    -- :add(SBatchNorm(frame[3]))
    :add(ReLU(true))

    :add(nn.View(-1):setNumInputDims(3))
    :add(nn.Linear(frame[3], 5))
    -- :add(nn.Concat(2)
    --     :add(nn.Sequential():add(nn.Linear(20,2)):add(nn.Tanh()))
    --     :add(nn.Sequential():add(nn.Linear(20,2)):add(nn.Sigmoid()))
    -- )

    :add(nn.Linear(5,1))
    :add(nn.Tanh())

-- weight initial
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
       local n = v.kW*v.kH*v.nOutputPlane
       v.weight:normal(0,math.sqrt(2/n))
       v.bias:zero()
    end
  end
  local function BNInit(name)
    for k,v in pairs(model:findModules(name)) do
       v.weight:fill(1)
       v.bias:zero()
    end
  end

  ConvInit('nn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')
  BNInit('nn.SpatialBatchNormalization')
  BNInit('nn.SpatialBatchNormalization')
  BNInit('nn.SpatialBatchNormalization')
  for k,v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
  end
  return model
end

return createModel
