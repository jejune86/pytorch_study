from torch import nn
import torch

class Net(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1,2), #linear layer
            nn.Sigmoid()
        )
        self.fc = nn.Linear(2, 1)
        self._init_weights() # 메소드 이름 앞에 언더바 하나면 protected
        
    def _init_weights(self) : 
        nn.init.xavier_uniform_(self.layer[0].weight) #Linear layer만 weight initialization
        self.layer[0].bias.data.fill_(0.01) # 편향은 상수값으로 
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill(0.01)
        
        
#가중치 초기화 메소드 모듈화
class Net(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1,2), #linear layer
            nn.Sigmoid()
        )
        self.fc = nn.Linear(2, 1)
        self.apply(self._init_weights)
               
    def _init_weights(self, module) : 
        if isinstance(module, nn.Linear) :
            nn.init.xavier_uniform_(module.wieght)
            nn.init.constant_(module.bias,0.01)
        print(f"Apply : {module}")
        self.layer[0].bias.data.fill_(0.01)

model = Net()

#가중치 초기화 함수

# 상수 초기화
torch.nn.init.constant_(tensor, val)

# 스칼라(1) 초기화 
torch.nn.init.ones_(tensor)

# 스칼라(0) 초기화
torch.nn.init.zeros_(tensor)

# 스칼라(eye) 초기화 (대각선을 1로 채우고 나머지 0, 2차원만 가능)
torch.nn.init.eye_(tensor)

# 디랙 델타 함수 초기화
torch.nn.init.dirac_(tensor)

#균등 분포 초기화
torch.nn.init.uniform_(tensor, a=0.0, b=1.0)

# 정규 분포 초기화
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)

# 잘린 정규 분포 초기화
torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)

# 희소 정규 분포 초기화
torch.nn.init.sparse_(tensor, sparsity=0.1, std=0.01)

# 제이비어 초기화 (균등 분포)
torch.nn.init.xavier_uniform_(tensor, gain=1.0)

# 제이비어 초기화 (정규 분포)
torch.nn.init.xavier_normal_(tensor, gain=1.0)

# 카이밍 초기화 (균등 분포)
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

# 카이밍 초기화 (정규 분포)
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

# 직교 초기화
torch.nn.init.orthogonal_(tensor, gain=1.0) # tensor를 직교 행렬로 초기화