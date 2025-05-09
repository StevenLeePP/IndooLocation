import torch
from torch import nn

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.flatten_size = 64 * 256 * 2  # Based on input (1024, 8) after pooling
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.BatchNorm1d(1024),  
            nn.ReLU(),
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # Input x shape: (batch_size, 1, 1024, 8)
        x = self.cnn(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x
