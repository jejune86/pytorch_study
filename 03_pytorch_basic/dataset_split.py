import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from custom import CustomDataset

dataset = CustomDataset("./datasets/non_linear.csv")
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(validation_dataset)}")
print(f"Test Data Size : {len(test_dataset)}")

#무작위 분리 함수
# subset = torch.utils.data.random_split(
#     dataset,
#     lengths, #데이터셋을 분할할 길이의 리스트
#     generator # 난수 생성 시드,  torch.maual_seed(int)로 고정 가능
# )
