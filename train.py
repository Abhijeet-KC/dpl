# train function && to call this script dataset and model import
import torch
import torch.nn as nn
import torch.optim as optim
from dpl import dataset
from dpl import model

epochs = 40
alpha = 1e-4
train_loader, _, _ = dataset.get_dataloaders(root="./data", batch_size=32, shuffle=True, transforms=dataset.transforms)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_input = model.get_model(in_channel=1, out_channel=10).to(device)
criterion = nn.CrossEntropyLoss()


def train(train_loader=train_loader, model=model_input, epochs=epochs, criterion=criterion, lr=alpha, device=device):
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in range(epochs):
        total_loss = 0
        for idx, data in enumerate(train_loader):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            
            # Forward pass
            y_pred = model(X)        
            
            loss = criterion(y_pred, y)
            total_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update
            
            optimizer.step()
            
        print(f"Loss: {total_loss/len(train_loader)}, for epoch {epoch+1}/{epochs} ")
    
if __name__ == "__main__":
    train()