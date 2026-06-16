import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(3,30)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(30,30)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(30,30)

        self.sigmoid3 = nn.Linear(30,18)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.linear3(x)

        x = self.sigmoid3(x)

        return x