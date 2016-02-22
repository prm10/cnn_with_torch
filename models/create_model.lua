require 'torch'
require 'nn'
function create_model()
  local model=require('models/cnn')()
  local criterion = nn.MSECriterion()
  return model,criterion
end
return create_model
