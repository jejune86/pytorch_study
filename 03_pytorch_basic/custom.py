import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset) :
    def __init__(self, file_path) :
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        # df.iloc[:,0].value -> 첫 열의 있는 데이터를 numpy 배열로 반환환
        self.y = df.iloc[:, 1].values
        self.length = len(df)
        
    def __getitem__(self, index) :
        x = torch.FloatTensor([self.x[index]**2, self.x[index]])
        # x 는 [x^2, x] 형태로 반환
        y = torch.FloatTensor([self.y[index]])
        return x, y
    
    def __len__(self) :
        return self.length
    
class CustomModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.layer = nn.Linear(2,1)
        
        
    def forward(self, x) :
        x = self.layer(x)
        return x
    

train_dataset = CustomDataset("./datasets/non_linear.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device) #to 로 어떤 장치로 연산을 진행할지 설정 가능
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10000) :
    cost = 0.0
    for x, y in train_dataloader :
        x = x.to(device)
        y = y.to(device)
        
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss
    
    cost = cost / len(train_dataloader)
    
    if (epoch + 1) % 1000 == 0 :
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")
        
# 모델 평가

with torch.no_grad() : # 기울기 계산을 비활성화화
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
    
    
# 모델 저장 
torch.save(
    model,
    "./models/model.pt"
)

torch.save(
    model.state.dict(),
    "./models/model_state_dict.pt"
)