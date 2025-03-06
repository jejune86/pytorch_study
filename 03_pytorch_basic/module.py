# module class는 __init__ 과 forward를 재정의하여 활용

# ex) 모듈 클래스 기본형
from torch import nn


# class Model(nn.Module) :
#     def __init__(self) :
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
        
#     def forward(self, x) :
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x)) 
#         return x