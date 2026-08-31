import torch
from torch import nn
from training import is_inclusion

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

class IPINN(nn.Module):
    def __init__(self):
        super(IPINN, self).__init__()

        self.linear1 = nn.Linear(3,30)
        self.tanh1 = nn.Tanh()
        self.silu1 = nn.SiLU()
        self.linear2 = nn.Linear(30,30)
        self.tanh2 = nn.Tanh()
        self.silu2 = nn.SiLU()
        self.linear3 = nn.Linear(30,18)

    def forward(self, x):
        is_incl = is_inclusion(x)
        is_incl = is_incl.unsqueeze(-1)
            
        x = self.linear1(x)
        x = torch.where(is_incl == True, self.tanh1(x), self.silu1(x))
        x = self.linear2(x)
        x = torch.where(is_incl == True, self.tanh2(x), self.silu2(x))
        x = self.linear3(x)

        return x
