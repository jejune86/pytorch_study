#가중치 감쇠(Weight Decay)
## 모델이 더 작은 가중치를 갖도록 손실함수에 규제

import torch

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
# pytorch의 가중치 감쇠는 L2정규화와 동일, weight_decay는 L2 정규화의 람다값    

# 모멘텀
# 이전 단계의 기울기를 참고하여 가중치를 업데이트하는 방법
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Elastic-Net L1과 L2를 합친 규제
