import pyvista as pv
import torch
from torch.utils.data import Dataset
import csv

class EshelbyDataset(Dataset):
    def __init__(self,points,u):
        self.points = points
        self.u = u

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index], self.u[index]

class EshelbyFullDataset(Dataset):
    def __init__(self,points,mat_params,indices,u):
        self.points = points
        self.mat_params = mat_params
        self.indices = indices
        self.u = u

    def __len__(self):
        return len(self.u)

    def __getitem__(self,index):
        return torch.tensor(self.points[int(self.indices[index][0])],dtype=torch.float32), self.mat_params[int(self.indices[index][1])], self.u[index]