import torch
from torch import nn
from torch.func import jacrev, vmap
from torch.utils.data import random_split, DataLoader
import numpy as np
import pyvista as pv
from sklearn.preprocessing import StandardScaler
import csv
import pickle
from model import Net
from dataset import EshelbyDataset

class Trainer:
    def __init__(self):
        self.simulation_file = "sim0001_alto.vtu"
        self.parameters_file = "parameter_overview.csv"
        
        self.test_set_ratio = 0.2
        self.is_PINN = True
        self.learning_rate = 5e-3
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Using {self.device} for training")
        
        self.mesh = pv.read(self.simulation_file)
        self.points = torch.tensor(self.mesh.points,dtype=torch.float32)

        with open(self.parameters_file) as f:
            parameters = csv.DictReader(f)
            for row in parameters:
                self.E_inclusion = float(row['E_inclusion'])
                self.nu_inclusion = float(row['nu_inclusion'])
                self.E_matrix = float(row['E_matrix'])
                self.nu_matrix = float(row['nu_matrix'])

        self.lambda_inclusion = self.E_inclusion*self.nu_inclusion/((1-2*self.nu_inclusion)*(1+self.nu_inclusion))
        self.mu_inclusion = self.E_inclusion/(2*(1+self.nu_inclusion))
        self.lambda_matrix = self.E_matrix*self.nu_matrix/((1-2*self.nu_matrix)*(1+self.nu_matrix))
        self.mu_matrix = self.E_matrix/(2*(1+self.nu_matrix))

        self.u_1 = torch.tensor(self.mesh.point_data["u_1"],dtype=torch.float32)
        #u_2 = torch.tensor(self.mesh.point_data["u_2"],dtype=torch.float32)
        #u_3 = torch.tensor(self.mesh.point_data["u_3"],dtype=torch.float32)
        #u_4 = torch.tensor(self.mesh.point_data["u_4"],dtype=torch.float32)
        #u_5 = torch.tensor(self.mesh.point_data["u_5"],dtype=torch.float32)
        #u_6 = torch.tensor(self.mesh.point_data["u_6"],dtype=torch.float32)

        #self.u = torch.cat([u_1,u_2,u_3,u_4,u_5,u_6],axis=-1)

        generator = torch.Generator().manual_seed(0)
        self.points_train, self.points_test = random_split(self.points,[1-self.test_set_ratio,self.test_set_ratio],generator=generator)
        generator = torch.Generator().manual_seed(0)
        self.u_train, self.u_test = random_split(self.u_1,[1-self.test_set_ratio,self.test_set_ratio],generator=generator)

        self.scaler = StandardScaler()
        self.scaler.fit(self.u_train)

        self.scale_tensor = torch.tensor(self.scaler.scale_,device=self.device)
        self.mean_tensor = torch.tensor(self.scaler.mean_,device=self.device)

        self.u_train_scaled = self.scaler.transform(self.u_train).astype(np.float32)
        self.u_test_scaled = self.scaler.transform(self.u_test).astype(np.float32)

        self.train_dataset = EshelbyDataset(self.points_train,self.u_train_scaled)
        self.test_dataset = EshelbyDataset(self.points_test,self.u_test_scaled)

        self.train_dataloader = DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=100)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=100)

        self.model = Net()
        self.model.to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

    def residual(self, X):
        def unscaled_output(X):
            return (self.model(X)*self.scale_tensor + self.mean_tensor)

        is_incl = is_inclusion(X) #check which points are in inclusion and which in matrix
        lam = torch.where(is_incl == True, self.lambda_inclusion, self.lambda_matrix)
        mu = torch.where(is_incl == True, self.mu_inclusion, self.mu_matrix)

        lam = lam.to(self.device)
        #lam = lam.unsqueeze(-1).expand(-1,6)
        mu = mu.to(self.device)
        #mu = mu.unsqueeze(-1).expand(-1,6)

        hessian = vmap(jacrev(jacrev(unscaled_output)))(X).squeeze()

        dux_dxdx = hessian[:,0,0,0]
        dux_dydy = hessian[:,0,1,1]
        dux_dzdz = hessian[:,0,2,2]
        dux_dydx = hessian[:,0,0,1]
        dux_dzdx = hessian[:,0,0,2]

        duy_dxdx = hessian[:,1,0,0]
        duy_dydy = hessian[:,1,1,1]
        duy_dzdz = hessian[:,1,2,2]
        duy_dxdy = hessian[:,1,1,0]
        duy_dzdy = hessian[:,1,1,2]

        duz_dxdx = hessian[:,2,0,0]
        duz_dydy = hessian[:,2,1,1]
        duz_dzdz = hessian[:,2,2,2]
        duz_dxdz = hessian[:,2,2,0]
        duz_dydz = hessian[:,2,2,1]

        return (lam+mu)*(dux_dxdx+duy_dxdy+duz_dxdz)+mu*(dux_dxdx+dux_dydy+dux_dzdz), \
                (lam+mu)*(dux_dydx+duy_dydy+duz_dydz)+mu*(duy_dxdx+duy_dydy+duy_dzdz), \
                (lam+mu)*(dux_dzdx+duy_dzdy+duz_dzdz)+mu*(duz_dxdx+duz_dydy+duz_dzdz)

    def train_update(self):
        loss_data_epoch = 0.0
        
        if self.is_PINN:
            point_pinn = (60*torch.rand(1000,3,device=self.device)-30)
            loss_pinn_epoch = 0.0
    
        for (point,u) in self.train_dataloader:
            point = point.to(self.device)
            u = u.to(self.device)
        
            pred = self.model(point)
            loss_data = self.loss_fn(pred,u)

            if self.is_PINN:
                res_x, res_y, res_z = self.residual(point_pinn)
                loss_pinn = self.loss_fn(res_x,torch.zeros_like(res_x)) + self.loss_fn(res_y,torch.zeros_like(res_y)) + self.loss_fn(res_z,torch.zeros_like(res_z))
                loss = loss_data + loss_pinn
            else:
                loss = loss_data
        
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss = loss.item()
        
            loss_data_epoch += loss_data.item()
            if self.is_PINN:
                loss_pinn_epoch += loss_pinn.item()

        loss_data_epoch /= len(self.train_dataloader)
        if self.is_PINN:
            loss_pinn_epoch /= len(self.train_dataloader)
        else:
            loss_pinn_epoch = None

        return loss_data_epoch, loss_pinn_epoch

    def train_loop(self, N):
        curves_file = open("loss_curves.dat","w")
        if self.is_PINN:
            curves_file.write("#epoch loss_data loss_pinn\n")
        else:
            curves_file.write("#epoch loss_data\n")

        for epoch in range(N):
            loss_epoch, loss_pinn = self.train_update()
            if self.is_PINN:
                curves_file.write(str(epoch+1) + " " + str(loss_epoch) + " " + str(loss_pinn) + "\n")
            else:
                curves_file.write(str(epoch+1) + " " + str(loss_epoch) + "\n")

        curves_file.close()

    def test_loop(self):
        loss = 0.0
        with torch.no_grad():
            f = open("pred.dat","w")
            f.write("#x y z u1x_true u1x_pred u1y_true u1y_pred u1z_true u1z_pred\n")
            for (point, u) in self.test_dataloader:
                point = point.to(self.device)
                u = u.to(self.device)

                pred = self.model(point)
                loss += self.loss_fn(pred,u).item()

                u = self.scaler.inverse_transform(u.cpu())
                pred = self.scaler.inverse_transform(pred.cpu())

                for i in range(len(point)):
                    f.write(str(point[i][0].item()) + " " + str(point[i][1].item()) + " " + str(point[i][2].item()) + " ")
                    for j, value in enumerate(pred[i]):
                        f.write(str(u[i][j].item()) + " " + str(value.item()) + " ")
                    f.write("\n")
            f.close()
        
        loss /= len(self.test_dataloader)
        return(loss)

    def save_weights(self, file_weights="weights.pt", file_scaler="scaler.pkl"):
        torch.save(self.model.state_dict(),file_weights)
        with open(file_scaler,"wb") as f:
            pickle.dump(self.scaler,f)

def is_inclusion(pos):
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    z_min, z_max = -1.0, 1.0

    inside_x = (pos[...,0] >= x_min) & (pos[...,0] <=x_max)
    inside_y = (pos[...,1] >= y_min) & (pos[...,1] <=y_max)
    inside_z = (pos[...,2] >= z_min) & (pos[...,2] <=z_max)

    return inside_x & inside_y & inside_z