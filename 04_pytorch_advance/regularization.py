import torch
from torch import nn



# L1 Regression AKA Lasso Regression (가중치 절대값의 합)
for x, y in train_data_loader:
    x = x.to(device)
    y = y.to(device)

    output = model(x) 
    _lambda = 0.5
    l1_loss = sum(p.abs().sum() for p in model.parameters())
    
    loss = ciriterion(output, y) + _lambda * l1_loss

# L2 Regression AKA Ridge Regression (가중치 제곱의 합)
for x, y in train_data_loader:
    x = x.to(device)
    y = y.to(device)

    output = model(x) 
    
    _lambda = 0.5
    l2_loss = sum(p.pow(2).sum() for p in model.parameters())
    
    loss = ciriterion(output, y) + _lambda * l2_loss
    
    # 비교
    #             L1                   L2 
    #특징 선택     O                    X
    #이상치       강함                 약함
    #가중치      0 가능            0에 가까워짐
    #학습     복잡한 데이터 x      복잡한 데이터 o
    