import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,256)
        self.layer2 = nn.Linear(256,1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x