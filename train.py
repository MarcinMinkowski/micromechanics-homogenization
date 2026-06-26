import pyvista as pv
import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from dataset import EshelbyDataset
import csv
from model import Net
import training

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    data = EshelbyDataset()

    train_dataset, test_dataset = random_split(data, [0.8,0.2])

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=100, num_workers=7)
    test_dataloader = DataLoader(dataset=test_dataset)

    model = Net()
    model.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(10):
        loss_epoch, loss_pinn = training.train_loop(data, model, dataloader, loss_fn, optimizer, device)
        #loss_epoch = training.train_loop(model, dataloader, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}: data loss: {loss_epoch}, PINN loss: {loss_pinn}")
        #print(f"Epoch {epoch+1}: data loss: {loss_epoch}")

    test_loss = training.test_loop(model, test_dataloader, loss_fn, device)
    print(f"Test set loss: {loss}")