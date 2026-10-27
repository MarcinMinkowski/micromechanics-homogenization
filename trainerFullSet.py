import torch
from torch import nn
from torch.func import jacrev, vmap
from torch.utils.data import random_split, DataLoader
import numpy as np
import pyvista as pv
import h5py
from sklearn.preprocessing import StandardScaler
import csv
import pickle
from model import Net, IPINN, IPINN_material
from dataset import EshelbyFullDataset
import copy
import os

class TrainerFullSet:
    def __init__(self):
        self.mesh_file = "mesh/mesh.vtu"
        self.samples_dir = "samples"
        self.disp_file = "displacement.h5"
        
        self.val_set_ratio = 0.1          #validation to whole dataset ratio
        self.test_set_ratio = 0.1         #test to whole dataset ratio
        self.is_PINN = False              #is PINN used while training
        self.learning_rate = 5e-3
        self.patience = 20
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Using {self.device} for training")
        
        self.mesh = pv.read(self.mesh_file)
        self.points = self.mesh.points                                #tensor with all points in the mesh

        point_index = torch.arange(len(self.points)).unsqueeze(1)     #tensor with indices for points in the mesh (same for each sample)
        #list_dir = os.listdir(self.samples_dir)                       #list of directories with samples
        list_dir = ["sid=00000","sid=00001"]
        n_samples = len(list_dir)                                     #number of samples
        
        #self.u_all = torch.empty(0,5)                                 #tensor with u coordinates + point index + sample index
        self.u_all = torch.empty(0,3)                                 #tensor with u coordinates
        self.indices_all = torch.empty(0,2)                           #tensor with point and sample indices
        self.mat_params = torch.empty(n_samples,4)                    #tensor with material parameters for each sample

        for dir in list_dir:
            file_name = os.path.join(self.samples_dir,dir,self.disp_file)
            with h5py.File(file_name,"r") as f:
                E_inclusion = f.attrs["E_inclusion"]
                nu_inclusion = f.attrs["nu_inclusion"]
        
                lambda_inclusion = nu_inclusion*E_inclusion/((1+nu_inclusion)*(1-2*nu_inclusion))
                mu_inclusion = E_inclusion/(2*(1+nu_inclusion))
        
                E_matrix = f.attrs["E_matrix"]
                nu_matrix = f.attrs["nu_matrix"]

                lambda_matrix = nu_matrix*E_matrix/((1+nu_matrix)*(1-2*nu_matrix))
                mu_matrix = E_matrix/(2*(1+nu_matrix))

                self.mat_params[f.attrs["sid"],0] = lambda_inclusion
                self.mat_params[f.attrs["sid"],1] = mu_inclusion
                self.mat_params[f.attrs["sid"],2] = lambda_matrix
                self.mat_params[f.attrs["sid"],3] = mu_matrix

                u = torch.tensor(f["u_total"][0])                                #displacement field for given sample for xx loading
                sample_index = torch.full((len(u),1),f.attrs["sid"])             #sample index
                indices = torch.cat((point_index,sample_index),1)        #point index + sample index
                
            self.u_all = torch.cat((self.u_all,u))
            self.indices_all = torch.cat((self.indices_all,indices))
        
        val_and_test_set_ratio = self.val_set_ratio + self.test_set_ratio                 #validation and test combined to whole dataset ratio
        val_to_val_and_test_ratio = self.val_set_ratio / val_and_test_set_ratio           #validation to combined validation and test ratio
        
        generator = torch.Generator().manual_seed(0)                             #splitting dataset into training and (val+test) sets
        self.u_train, self.u_val_and_test = random_split(self.u_all,[1-val_and_test_set_ratio,val_and_test_set_ratio],generator=generator)
        generator = torch.Generator().manual_seed(0)
        self.indices_train, self.indices_val_and_test = random_split(self.indices_all,[1-val_and_test_set_ratio,val_and_test_set_ratio],generator=generator)

        generator = torch.Generator().manual_seed(0)                             #splitting (val+test) into val and test sets
        self.u_val, self.u_test = random_split(self.u_val_and_test,[val_to_val_and_test_ratio,1-val_to_val_and_test_ratio],generator=generator)
        generator = torch.Generator().manual_seed(0)
        self.indices_val, self.indices_test = random_split(self.indices_val_and_test,[val_to_val_and_test_ratio,1-val_to_val_and_test_ratio],generator=generator)

        self.scaler = StandardScaler()
        self.scaler.fit(self.u_train)

        self.scale_tensor = torch.tensor(self.scaler.scale_,device=self.device)
        self.mean_tensor = torch.tensor(self.scaler.mean_,device=self.device)

        self.u_train_scaled = self.scaler.transform(self.u_train).astype(np.float32)
        self.u_val_scaled = self.scaler.transform(self.u_val).astype(np.float32)
        self.u_test_scaled = self.scaler.transform(self.u_test).astype(np.float32)

        self.train_dataset = EshelbyFullDataset(self.points,self.mat_params,self.indices_train,self.u_train_scaled)
        self.val_dataset = EshelbyFullDataset(self.points,self.mat_params,self.indices_val,self.u_val_scaled)
        self.test_dataset = EshelbyFullDataset(self.points,self.mat_params,self.indices_test,self.u_test_scaled)

        self.train_dataloader = DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=100)
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=100)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=100)

        self.model = IPINN_material()
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

    def interface_points(self, n_points, direction, value):
        points = 2*torch.rand(n_points,3,device=self.device)-1.0
        points[:,direction] = value
        return points

    def get_stress_tensor(self, X):
        def unscaled_inclusion_output(X):
            return (self.model.inclusion(X)*self.scale_tensor + self.mean_tensor)
        def unscaled_matrix_output(X):
            return (self.model.matrix(X)*self.scale_tensor + self.mean_tensor)
        
        jacobian_inclusion = vmap(jacrev(unscaled_inclusion_output))(X)
        jacobian_matrix = vmap(jacrev(unscaled_matrix_output))(X)

        strain_inclusion = (jacobian_inclusion+jacobian_inclusion.mT)/2
        strain_matrix = (jacobian_matrix+jacobian_matrix.mT)/2

        strain_inclusion_trace = torch.diagonal(strain_inclusion,dim1=-2,dim2=-1).sum(1).unsqueeze(-1).unsqueeze(-1)
        strain_matrix_trace = torch.diagonal(strain_matrix,dim1=-2,dim2=-1).sum(1).unsqueeze(-1).unsqueeze(-1)

        stress_inclusion = self.lambda_inclusion*torch.eye(3,device=self.device)*strain_inclusion_trace+2*self.mu_inclusion*strain_inclusion
        stress_matrix = self.lambda_matrix*torch.eye(3,device=self.device)*strain_matrix_trace+2*self.mu_matrix*strain_matrix

        return stress_inclusion, stress_matrix

    def train_update(self):
        loss_data_epoch = 0.0
        
        if self.is_PINN:
            point_pinn = (60*torch.rand(1000,3,device=self.device)-30) #collocation points
            
            point_interface_xl = self.interface_points(100,0,-1)             #interface points
            point_interface_xh = self.interface_points(100,0,1)
            point_interface_yl = self.interface_points(100,1,-1)
            point_interface_yh = self.interface_points(100,1,1)
            point_interface_zl = self.interface_points(100,2,-1)
            point_interface_zh = self.interface_points(100,2,1)
            point_interface = torch.cat([point_interface_xl,point_interface_xh,point_interface_yl,point_interface_yh,point_interface_zl,point_interface_zh])
            
            loss_pinn_epoch = 0.0
            loss_interface_epoch = 0.0
            loss_traction_epoch = 0.0
    
        for (point,mat_params,u) in self.train_dataloader:
            point = point.to(self.device)
            mat_params = mat_params.to(self.device)
            u = u.to(self.device)

            input = torch.cat([point,mat_params],1)
        
            pred = self.model(input)
            loss_data = self.loss_fn(pred,u)

            if self.is_PINN:
                res_x, res_y, res_z = self.residual(point_pinn) #residual
                loss_pinn = self.loss_fn(res_x,torch.zeros_like(res_x)) + self.loss_fn(res_y,torch.zeros_like(res_y)) + self.loss_fn(res_z,torch.zeros_like(res_z))

                loss_interface = self.loss_fn(self.model.matrix(point_interface),self.model.inclusion(point_interface))

                stress_inclusion_x, stress_matrix_x = self.get_stress_tensor(torch.cat([point_interface_xl,point_interface_xh]))
                stress_inclusion_y, stress_matrix_y = self.get_stress_tensor(torch.cat([point_interface_yl,point_interface_yh]))
                stress_inclusion_z, stress_matrix_z = self.get_stress_tensor(torch.cat([point_interface_zl,point_interface_zh]))

                traction_inclusion_x = torch.matmul(stress_inclusion_x,torch.tensor([1.0,0.0,0.0],device=self.device))
                traction_matrix_x = torch.matmul(stress_matrix_x,torch.tensor([1.0,0.0,0.0],device=self.device))
                traction_inclusion_y = torch.matmul(stress_inclusion_y,torch.tensor([0.0,1.0,0.0],device=self.device))
                traction_matrix_y = torch.matmul(stress_matrix_y,torch.tensor([0.0,1.0,0.0],device=self.device))
                traction_inclusion_z = torch.matmul(stress_inclusion_z,torch.tensor([0.0,0.0,1.0],device=self.device))
                traction_matrix_z = torch.matmul(stress_matrix_z,torch.tensor([0.0,0.0,1.0],device=self.device))

                loss_traction = (self.loss_fn(traction_inclusion_x,traction_matrix_x) + self.loss_fn(traction_inclusion_y,traction_matrix_y) + self.loss_fn(traction_inclusion_z,traction_matrix_z))/3
                
                loss = loss_data + loss_pinn + loss_interface + loss_traction
            else:
                loss = loss_data
        
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss = loss.item()
        
            loss_data_epoch += loss_data.item()
            if self.is_PINN:
                loss_pinn_epoch += loss_pinn.item()
                loss_interface_epoch += loss_interface.item()
                loss_traction_epoch += loss_traction.item()

        loss_data_epoch /= len(self.train_dataloader)
        if self.is_PINN:
            loss_pinn_epoch /= len(self.train_dataloader)
            loss_interface_epoch /= len(self.train_dataloader)
            loss_traction_epoch /= len(self.train_dataloader)
        else:
            loss_pinn_epoch = None
            loss_interface_epoch = None
            loss_traction_epoch = None

        return loss_data_epoch, loss_pinn_epoch, loss_interface_epoch, loss_traction_epoch

    def train_loop(self, N):
        curves_file = open("loss_curves.dat","w")
        if self.is_PINN:
            curves_file.write("#epoch loss_data loss_pinn loss_interface loss_traction val_loss_data\n")
        else:
            curves_file.write("#epoch loss_data val_loss_data\n")

        best_val_loss = float('inf')
        best_model_weights = None

        for epoch in range(N):
            print(f"Epoch {epoch+1} / {N}")
            print("----------------")
            loss_epoch, loss_pinn, loss_interface, loss_traction = self.train_update()
            val_loss = self.val_loop()
            print(f"Training data loss: {loss_epoch}")
            print(f"Validation data loss: {val_loss}")
            if self.is_PINN:
                curves_file.write(str(epoch+1) + " " + str(loss_epoch) + " " + str(loss_pinn) + " " + str(loss_interface) + " " + str(loss_traction) + " " + str(val_loss) + "\n")
            else:
                curves_file.write(str(epoch+1) + " " + str(loss_epoch) + " " + str(val_loss) + "\n")

            if val_loss < best_val_loss:
                patience_current = 0
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                patience_current += 1
                print(f"No validation loss improvement for {patience_current} / {self.patience} epoch(s)")
                if patience_current == self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        curves_file.close()
        self.model.load_state_dict(best_model_weights)

    def val_loop(self):
        loss = 0.0
        with torch.no_grad():
            for (point,mat_params,u) in self.val_dataloader:
                point = point.to(self.device)
                mat_params = mat_params.to(self.device)
                u = u.to(self.device)

                input = torch.cat([point,mat_params],1)

                pred = self.model(input)
                loss += self.loss_fn(pred,u).item()

        loss /= len(self.val_dataloader)
        return(loss)

    def test_loop(self):
        loss = 0.0
        with torch.no_grad():
            f = open("pred.dat","w")
            f.write("#x y z u1x_true u1x_pred u1y_true u1y_pred u1z_true u1z_pred\n")
            for (point,mat_params,u) in self.test_dataloader:
                point = point.to(self.device)
                mat_params = mat_params.to(self.device)
                u = u.to(self.device)

                input = torch.cat([point,mat_params],1)

                pred = self.model(input)
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
