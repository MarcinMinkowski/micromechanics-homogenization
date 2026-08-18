import pyvista as pv
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from dataset import EshelbyDataset
import csv
from model import Net
import training

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    mesh = pv.read("sim0001_alto.vtu")

    points = torch.tensor(mesh.points,dtype=torch.float32)

    u_1 = torch.tensor(mesh.point_data["u_1"],dtype=torch.float32)
    u_2 = torch.tensor(mesh.point_data["u_2"],dtype=torch.float32)
    u_3 = torch.tensor(mesh.point_data["u_3"],dtype=torch.float32)
    u_4 = torch.tensor(mesh.point_data["u_4"],dtype=torch.float32)
    u_5 = torch.tensor(mesh.point_data["u_5"],dtype=torch.float32)
    u_6 = torch.tensor(mesh.point_data["u_6"],dtype=torch.float32)

    u = torch.cat([u_1,u_2,u_3,u_4,u_5,u_6],axis=-1)

    generator = torch.Generator().manual_seed(0)
    points_train, points_test = random_split(points,[0.8,0.2],generator=generator)
    generator = torch.Generator().manual_seed(0)
    u_train, u_test = random_split(u,[0.8,0.2],generator=generator)

    scaler = StandardScaler()
    scaler.fit(u_train)

    u_train_scaled = scaler.transform(u_train).astype(np.float32)
    u_test_scaled = scaler.transform(u_test).astype(np.float32)

    train_dataset = EshelbyDataset(mesh,points_train,u_train_scaled)
    test_dataset = EshelbyDataset(mesh,points_test,u_test_scaled)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=100)
    test_dataloader = DataLoader(dataset=test_dataset)

    model = Net()
    model.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    is_PINN = True

    curves_file = open("loss_curves.dat","w")
    curves_file.write("#epoch loss_data loss_pinn\n")

    for epoch in range(100):
        loss_epoch, loss_pinn = training.train_loop(train_dataset, model, train_dataloader, loss_fn, optimizer, is_PINN, scaler, device)
        if loss_epoch:
            curves_file.write(str(epoch+1) + " " + str(loss_epoch) + " " + str(loss_pinn) + "\n")
        else:
            curves_file.write(str(epoch+1) + " " + str(loss_epoch) + "\n")

    curves_file.close()

    test_loss = training.test_loop(model, test_dataloader, loss_fn, scaler, device)

    with open("loss_test.dat","w") as f:
        f.write("Test loss: " + str(test_loss))
