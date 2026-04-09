# get model function, to return the model
import torch
import torch.nn as nn
from torch.nn import functional as F

class Classifer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=32*4*4, out_features=64),
            nn.Tanh(),
            nn.Dropout(),
            
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(),
            
            nn.Linear(32, out_channel)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.shape[0], -1) # similar to x.view(x.size(0), -1) or nn.Flatten() basically used for keeping B from B, C, H, W
        x = self.classifier(x)
        return F.softmax(x, dim=1)
    
def get_model(in_channel, out_channel):
    
    model = Classifer(in_channel, out_channel)
    
    return model