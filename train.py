import pyvista as pv
import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import EshelbyDataset
import csv
from model import Net
import training

data = EshelbyDataset()

dataloader = DataLoader(dataset=data, shuffle=True, batch_size=1)

model = Net()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(10):
    loss_epoch, loss_pinn = training.train_loop(model, dataloader, loss_fn, optimizer)
    print(f"Epoch {epoch+1}: data loss: {loss_epoch}, PINN loss: {loss_pinn}")