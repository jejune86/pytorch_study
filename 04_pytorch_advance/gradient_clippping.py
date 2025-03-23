import torch


grad_norm = torch.nn.utils.clip_grad_norm_(
    parameters,
    max_norm,
    norm_type=2.0
)

for x, y in train_data_loader:
    x = x.to(device)
    y = y.to(device)

    output = moedele(x)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    
    optimizer.step()