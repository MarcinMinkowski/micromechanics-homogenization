import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(3,30)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(30,30)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(30,18)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        x = self.linear3(x)

        return x
