requre 'nn'

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
  model = nn.Sequential()
    :add(Convolution(1,5,1,20,1,1,0,0))
    :add(SBatchNorm(5))
    :add(ReLU(true))
    :add(Max(1,10,1,3))

    :add(Convolution(5,10,1,20,1,1,0,0))
    :add(SBatchNorm(10))
    :add(ReLU(true))
    :add(Max(1,10,1,3))

    :add(Convolution(10,20,1,20,1,1,0,0))
    :add(SBatchNorm(20))
    :add(ReLU(true))
    :add(Max(1,10,1,3))

    :add(Convolution(20,30,1,20,1,1,0,0))
    :add(SBatchNorm(30))
    :add(ReLU(true))
    :add(Max(1,10,1,3))

    :add(Convolution(30,40,25,9,1,1,0,0))
    :add(SBatchNorm(40))
    :add(ReLU(true))

    :add(nn.View(-1):setNumInputDims(3))
    :add(nn.Linear(40, 20))
    :add(nn.Concat(2)
        :add(nn.Sequential():add(nn.Linear(20,2)):add(nn.Tanh()))
        :add(nn.Sequential():add(nn.Linear(20,2)):add(nn.Sigmoid()))
    )

-- weight initial
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
       local n = v.kW*v.kH*v.nOutputPlane
       v.weight:normal(0,math.sqrt(2/n))
       if nn.version >= 4000 then
          v.bias = nil
          v.gradBias = nil
       else
          v.bias:zero()
       end
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
