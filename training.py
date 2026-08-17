import torch
from torch.func import jacrev, vmap

def is_inclusion(pos):
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    z_min, z_max = -1.0, 1.0

    inside_x = (pos[:,0] >= x_min) & (pos[:,0] <=x_max)
    inside_y = (pos[:,0] >= y_min) & (pos[:,0] <=y_max)
    inside_z = (pos[:,0] >= z_min) & (pos[:,0] <=z_max)

    return inside_x & inside_y & inside_z

def residual(model, X, scale_tensor, mean_tensor, lam, mu):

    def unscaled_output(X):
        return (model(X)*scale_tensor + mean_tensor).view(-1,6,3)

    hessian = vmap(jacrev(jacrev(unscaled_output)))(X).squeeze()

    dux_dxdx = hessian[:,:,0,0,0]
    dux_dydy = hessian[:,:,0,1,1]
    dux_dzdz = hessian[:,:,0,2,2]
    dux_dxdy = hessian[:,:,0,0,1]
    dux_dxdz = hessian[:,:,0,0,2]

    duy_dxdx = hessian[:,:,1,0,0]
    duy_dydy = hessian[:,:,1,1,1]
    duy_dzdz = hessian[:,:,1,2,2]
    duy_dydx = hessian[:,:,1,1,0]
    duy_dydz = hessian[:,:,1,1,2]

    duz_dxdx = hessian[:,:,2,0,0]
    duz_dydy = hessian[:,:,2,1,1]
    duz_dzdz = hessian[:,:,2,2,2]
    duz_dzdx = hessian[:,:,2,2,0]
    duz_dzdy = hessian[:,:,2,2,1]

    return (lam+mu)*(dux_dxdx+duy_dydx+duz_dzdx)+mu*((dux_dxdx+dux_dydy+dux_dzdz)), \
            (lam+mu)*(dux_dxdy+duy_dydy+duz_dzdy)+mu*((duy_dxdx+duy_dydy+duy_dzdz)), \
            (lam+mu)*(dux_dxdz+duy_dydz+duz_dzdz)+mu*((duz_dxdx+duz_dydy+duz_dzdz))

def train_loop(data, model, dataloader, loss_fn, optimizer, is_PINN, scaler, device):
    loss_data_epoch = 0.0
    if is_PINN:
        loss_pinn_epoch = 0.0
        scale_tensor = torch.tensor(scaler.scale_,device=device)
        mean_tensor = torch.tensor(scaler.mean_,device=device)
    
    for (point,u) in dataloader:
        point = point.to(device)

        u = u.to(device)
        
        pred = model(point)
        loss_data = loss_fn(pred,u)
        

        if is_PINN:
            point_pinn = (60*torch.rand(1000,3,device=device)-30).requires_grad_(True)        #random points at which derivates for Navier-Cauchy equation are obtained

            is_incl = is_inclusion(point_pinn) #check which points are in inclusion and which in matrix
            lam = torch.where(is_incl == True, data.lambda_inclusion, data.lambda_matrix, device=device)
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
