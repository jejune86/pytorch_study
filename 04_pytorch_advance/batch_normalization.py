import torch
from torch import nn

x = torch.FloatTensor(
    [
        [-0.6577, -0.5797, 0.6360],
        [0.7392, 0.21445, 1.523],
        [0.2432, 0.5662, 0.322]
    ]
)

print(nn.BatchNorm1d(3)(x)) #3은 feature 수 
# 1d는 2D/3D 입력 데이터에 뱇치 정규화 수행

#Layer Normalization
## torch.nn.LayerNorm(normalized_shape)

#Instance Normalization
## torch.nn.InstanceNorm1d(num_features)  