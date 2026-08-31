import torch
from torch.func import jacrev, vmap

def is_inclusion(pos):
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    z_min, z_max = -1.0, 1.0

    inside_x = (pos[...,0] >= x_min) & (pos[...,0] <=x_max)
    inside_y = (pos[...,1] >= y_min) & (pos[...,1] <=y_max)
    inside_z = (pos[...,2] >= z_min) & (pos[...,2] <=z_max)

    return inside_x & inside_y & inside_z

def random_points(n, dist, device):
    points = torch.empty((0,3),device=device)

    while len(points)<n:
        n_new_points = n - len(points)

        new_points = (60*torch.rand(1,3,device=device)-30)

        matrix_x = (new_points[:,0] >= 1.0 + dist) | (new_points[:,0] <= -(1.0 + dist))
        matrix_y = (new_points[:,1] >= 1.0 + dist) | (new_points[:,1] <= -(1.0 + dist))
        matrix_z = (new_points[:,2] >= 1.0 + dist) | (new_points[:,2] <= -(1.0 + dist))

        matrix = matrix_x | matrix_y | matrix_z

        inclusion_x = (new_points[:,0] <= 1.0 - dist) & (new_points[:,0] >= -(1.0 - dist))
        inclusion_y = (new_points[:,1] <= 1.0 - dist) & (new_points[:,1] >= -(1.0 - dist))
        inclusion_z = (new_points[:,2] <= 1.0 - dist) & (new_points[:,2] >= -(1.0 - dist))

        inclusion = inclusion_x & inclusion_y & inclusion_z

        is_inside = matrix | inclusion
        
        new_points = new_points[is_inside]

        points = torch.cat([points,new_points],dim=0)

    return points

def residual(model, X, scale_tensor, mean_tensor, lam, mu):

    def unscaled_output(X):
        return (model(X)*scale_tensor + mean_tensor).view(-1,6,3)

    hessian = vmap(jacrev(jacrev(unscaled_output)))(X).squeeze()

    dux_dxdx = hessian[:,:,0,0,0]
    dux_dydy = hessian[:,:,0,1,1]
    dux_dzdz = hessian[:,:,0,2,2]
    dux_dydx = hessian[:,:,0,0,1]
    dux_dzdx = hessian[:,:,0,0,2]

    duy_dxdx = hessian[:,:,1,0,0]
    duy_dydy = hessian[:,:,1,1,1]
    duy_dzdz = hessian[:,:,1,2,2]
    duy_dxdy = hessian[:,:,1,1,0]
    duy_dzdy = hessian[:,:,1,1,2]

    duz_dxdx = hessian[:,:,2,0,0]
    duz_dydy = hessian[:,:,2,1,1]
    duz_dzdz = hessian[:,:,2,2,2]
    duz_dxdz = hessian[:,:,2,2,0]
    duz_dydz = hessian[:,:,2,2,1]

    return (lam+mu)*(dux_dxdx+duy_dxdy+duz_dxdz)+mu*(dux_dxdx+dux_dydy+dux_dzdz), \
            (lam+mu)*(dux_dydx+duy_dydy+duz_dydz)+mu*(duy_dxdx+duy_dydy+duy_dzdz), \
            (lam+mu)*(dux_dzdx+duy_dzdy+duz_dzdz)+mu*(duz_dxdx+duz_dydy+duz_dzdz)

def train_loop(data, model, dataloader, loss_fn, optimizer, is_PINN, scaler, device):
    loss_data_epoch = 0.0
    if is_PINN:
        #point_pinn = (60*torch.rand(1000,3,device=device)-30).requires_grad_(True)        #random points at which derivatives for Navier-Cauchy equation are obtained
        #point_pinn = random_points(1000,0.1,device).requires_grad_(True)
        point_pinn = random_points(1000,0.1,device)

        loss_pinn_epoch = 0.0
        scale_tensor = torch.tensor(scaler.scale_,device=device)
        mean_tensor = torch.tensor(scaler.mean_,device=device)
    
    for (point,u) in dataloader:
        point = point.to(device)

        u = u.to(device)
        
        pred = model(point)
        loss_data = loss_fn(pred,u)
        

        if is_PINN: 
            is_incl = is_inclusion(point_pinn) #check which points are in inclusion and which in matrix
            lam = torch.where(is_incl == True, data.lambda_inclusion, data.lambda_matrix)
            mu = torch.where(is_incl == True, data.mu_inclusion, data.mu_matrix)

            lam = lam.to(device)
            lam = lam.unsqueeze(-1).expand(-1,6)
            mu = mu.to(device)
            mu = mu.unsqueeze(-1).expand(-1,6)

            res_x, res_y, res_z = residual(model, point_pinn, scale_tensor, mean_tensor, lam, mu)

            loss_pinn = loss_fn(res_x,torch.zeros_like(res_x)) + loss_fn(res_y,torch.zeros_like(res_y)) + loss_fn(res_z,torch.zeros_like(res_z))

            loss = loss_data + loss_pinn
        else:
            loss = loss_data
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        
        loss_data_epoch += loss_data.item()
        if is_PINN:
            loss_pinn_epoch += loss_pinn.item()

    loss_data_epoch /= len(dataloader)
    if is_PINN:
        loss_pinn_epoch /= len(dataloader)
    else:
        loss_pinn_epoch = None

    return loss_data_epoch, loss_pinn_epoch

def test_loop(model, dataloader, loss_fn, scaler, device):
    loss = 0.0
    with torch.no_grad():
        f = open("pred.dat","w")
        f.write("#u1x_true u1x_pred u1y_true u1y_pred u1z_true u1z_pred u2x_true u2x_pred u2y_true u2y_pred u2z_true u2z_pred u3x_true u3x_pred u3y_true u3y_pred u3z_true u3z_pred u4x_true u4x_pred u4y_true u4y_pred u4z_true u4z_pred u5x_true u5x_pred u5y_true u5y_pred u5z_true u5z_pred u6x_true u6x_pred u6y_true u6y_pred u6z_true u6z_pred\n")
        for (point, u) in dataloader:
            point = point.to(device)

            u = u.to(device)

            pred = model(point)
            loss += loss_fn(pred,u).item()

            u = scaler.inverse_transform(u.cpu())
            pred = scaler.inverse_transform(pred.cpu())

            for i, value in enumerate(pred[0]):
                f.write(str(u[0][i].item()) + " " + str(value.item()) + " ")
            f.write("\n")
        f.close()
    loss /= len(dataloader)
    return(loss)
