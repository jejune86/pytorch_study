#Tensor : numpy의 ndarray와 유사한 구조로 배열이나 행렬과 유사한 것

import torch

#print(torch.tensor([1,2,3]))
print(torch.Tensor([[1,2,3],[4,5,6]]))
# 밑에는 Torch 클래스를 상속받은 것으로 미리 data type이 선언된 클래스 (Int, Double, Boolean 도 있음)
print(torch.LongTensor([1,2,3]))
print(torch.FloatTensor([1,2,3]))

# tensor attribute
tensor = torch.rand(1,2) #1x2 크기의 0~1사이의 난수값을 가진 tensor 생성
print(tensor)
print(tensor.shape)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

tensor = tensor.reshape(3,1) #차원 변환환