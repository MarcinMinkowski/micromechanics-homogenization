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
