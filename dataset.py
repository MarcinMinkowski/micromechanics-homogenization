import pyvista as pv
import torch
from torch.utils.data import Dataset
import csv

class EshelbyDataset(Dataset):
    def __init__(self):
        self.mesh = pv.read("sim0001_alto.vtu")
        self.u_1 = torch.tensor(self.mesh.point_data["u_1"],dtype=torch.float32)
        self.u_2 = torch.tensor(self.mesh.point_data["u_2"],dtype=torch.float32)
        self.u_3 = torch.tensor(self.mesh.point_data["u_3"],dtype=torch.float32)
        self.u_4 = torch.tensor(self.mesh.point_data["u_4"],dtype=torch.float32)
        self.u_5 = torch.tensor(self.mesh.point_data["u_5"],dtype=torch.float32)
        self.u_6 = torch.tensor(self.mesh.point_data["u_6"],dtype=torch.float32)
        #self.is_inclusion_pt = self.mesh.cell_data_to_point_data().point_data["inclusion"].astype(bool)

        with open("parameter_overview.csv") as f:
            parameters = csv.DictReader(f)
            for row in parameters:
                E_inclusion = float(row['E_inclusion'])
                nu_inclusion = float(row['nu_inclusion'])
                E_matrix = float(row['E_matrix'])
                nu_matrix = float(row['nu_matrix'])

        self.lambda_inclusion = E_inclusion*nu_inclusion/((1-2*nu_inclusion)*(1+nu_inclusion))
        self.mu_inclusion = E_inclusion/(2*(1+nu_inclusion))
        self.lambda_matrix = E_matrix*nu_matrix/((1-2*nu_matrix)*(1+nu_matrix))
        self.mu_matrix = E_matrix/(2*(1+nu_matrix))

    def __len__(self):
        return len(self.mesh.points)

    def __getitem__(self, index):
        return torch.tensor(self.mesh.points[index],dtype=torch.float32), torch.cat([self.u_1[index],self.u_2[index],self.u_3[index],self.u_4[index],self.u_5[index],self.u_6[index]])

    def is_inclusion(self, pos):
        return self.mesh.cell_data["inclusion"][self.mesh.find_containing_cell([pos[0],pos[1],pos[2]])].astype(bool)
