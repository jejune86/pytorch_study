#dataset은 클래스 형태로 제공됨

# __init__ : 초기화 메서드
# __getitem__ : 호출 메서드
# __len__ : 길이 반환


# Dataloader : dataset에 저장된 데이터를 불러오는 역할, batch_size, shuffle, num_workers 등 지정 가능

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

train_x = torch.FloatTensor([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])
train_y = torch.FloatTensor([
    [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
])

train_dataset = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

model = nn.Linear(2, 2, bias=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20000) :
    cost = 0.0
    for batch in train_dataloader :
        x, y = batch
        output = model(x)
        
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss
        
    cost = cost / len(train_dataloader)
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")