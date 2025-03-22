import torch
from torch import nn
#모델 저장
# torch.save(model, "./model.pth")

#모델 불러오기
# model = torch.load("./model.pth", map_loctaion) 
# 경로에서 모델 불러와서 map_location에 따라 cpu나 gpu로 연산할 수 있음


class CustomModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.layer = nn.Linear(2,1)
        
        
    def forward(self, x) :
        x = self.layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("./models/model.pt", map_location=device)
print(model)

with torch.no_grad() :
    model.eval()
    inputs = torch.FloatTensor(
        [
            [1 ** 2, 1],
            [5 ** 2, 5],
            [11 ** 2, 11]
        ]
    ).to(device)
    outputs = model(inputs)
    print(outputs)
    
    
#모델의 상태 저장/불러오기
# torch.save(
    # model.state_dict(),
    #"./model_state_dict.pth"
#) #모델에서 학습이 가능한 매개변수를 순서기 있는 딕셔너리 형식으로 반환

model_state_dict = torch.load("./models/model_state_dict.pth", map_location=device)
model.load_state_dict(model_state_dict)


#체크포인트 저장/불러오기

#저장장
checkpoint = 1
for epoch in range(10000) :
    
    cost = cost / len(train_dataloader)
    if (epoch+1) % 1000 == 0 :
        torch.save(
            {
                "model" : "CustomModel",
                "epoch" : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "cost" : cost,
                "description" : f"CustomModel 체크포인트-{checkpoint}",
            },
            f"./models/checkpoint-{checkpoint}.pth"
        )
        checkpoint += 1

#불러오기
checkpoint = torch.load("./models/checkpoint-1.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
checkpoint_epoch = checkpoint["epoch"]
checkpoint_description = checkpoint["description"]
print(checkpoint_description)


