import torch
from torch import optim

x = torch.Tensor(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)
y = torch.Tensor(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)
weight = torch.zeros(1, requires_grad=True) # 0 값을 갖는 tensor 생성, 크기는 1
bias = torch.zeros(1, requires_grad=True) 
learning_rate = 0.001

optimizer = optim.SGD([weight, bias], lr=learning_rate) #SGD optimizer 생성 (Stochastic Gradient Descent)

for epoch in range(10000):
    hypothesis = x * weight + bias # 가설, prediction
    cost = torch.mean((hypothesis - y) ** 2) # 손실함수
    
    optimizer.zero_grad()  # 이전 단계에서 계산된 기울기를 0으로 초기화
                           # PyTorch는 기울기를 누적시키기 때문에, 
                           # 매번 새로운 기울기를 계산하기 위해 초기화가 필요함
    cost.backward()        # 비용 함수를 미분하여 optimizer의 weight, bias 기울기 계산
    optimizer.step()       # 계산된 기울기를 사용하여 가중치와 편향 업데이트
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost.item():.3f}")