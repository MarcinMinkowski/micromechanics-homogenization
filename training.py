import torch

def residual(u, X, lam, mu):

    du1x = torch.autograd.grad(outputs=u[:,0],inputs=X,grad_outputs=torch.ones_like(u[:,0]),retain_graph=True,create_graph=True)[0]
    du1y = torch.autograd.grad(outputs=u[:,1],inputs=X,grad_outputs=torch.ones_like(u[:,1]),retain_graph=True,create_graph=True)[0]
    du1z = torch.autograd.grad(outputs=u[:,2],inputs=X,grad_outputs=torch.ones_like(u[:,2]),retain_graph=True,create_graph=True)[0]

    du1x_dx, du1x_dy, du1x_dz = du1x[:,0], du1x[:,1], du1x[:,2]
    du1y_dx, du1y_dy, du1y_dz = du1y[:,0], du1y[:,1], du1y[:,2]
    du1z_dx, du1z_dy, du1z_dz = du1z[:,0], du1z[:,1], du1z[:,2]

    u1_div = du1x_dx + du1y_dy + du1z_dz

    du1_div_d = torch.autograd.grad(outputs=u1_div,inputs=X,grad_outputs=torch.ones_like(u1_div),retain_graph=True)[0]

    du1x_dxd = torch.autograd.grad(outputs=du1x_dx,inputs=X,grad_outputs=torch.ones_like(du1x_dx),retain_graph=True)[0]
    du1x_dyd = torch.autograd.grad(outputs=du1x_dy,inputs=X,grad_outputs=torch.ones_like(du1x_dy),retain_graph=True)[0]
    du1x_dzd = torch.autograd.grad(outputs=du1x_dz,inputs=X,grad_outputs=torch.ones_like(du1x_dz),retain_graph=True)[0]

    u1x_laplacian = du1x_dxd[:,0] + du1x_dyd[:,1] + du1x_dzd[:,2]
        
    du1y_dxd = torch.autograd.grad(outputs=du1y_dx,inputs=X,grad_outputs=torch.ones_like(du1y_dx),retain_graph=True)[0]
    du1y_dyd = torch.autograd.grad(outputs=du1y_dy,inputs=X,grad_outputs=torch.ones_like(du1y_dy),retain_graph=True)[0]
    du1y_dzd = torch.autograd.grad(outputs=du1y_dz,inputs=X,grad_outputs=torch.ones_like(du1y_dz),retain_graph=True)[0]

    u1y_laplacian = du1y_dxd[:,0] + du1y_dyd[:,1] + du1y_dzd[:,2]

    du1z_dxd = torch.autograd.grad(outputs=du1z_dx,inputs=X,grad_outputs=torch.ones_like(du1z_dx),retain_graph=True)[0]
    du1z_dyd = torch.autograd.grad(outputs=du1z_dy,inputs=X,grad_outputs=torch.ones_like(du1z_dy),retain_graph=True)[0]
    du1z_dzd = torch.autograd.grad(outputs=du1z_dz,inputs=X,grad_outputs=torch.ones_like(du1z_dz),retain_graph=True)[0]

    u1z_laplacian = du1z_dxd[:,0] + du1z_dyd[:,1] + du1z_dzd[:,2]

    return (lam+mu)*du1_div_d[:,0]+mu*u1x_laplacian, (lam+mu)*du1_div_d[:,1]+mu*u1y_laplacian, (lam+mu)*du1_div_d[:,2]+mu*u1z_laplacian

def train_loop(data, model, dataloader, loss_fn, optimizer, device):
    loss_data_epoch = 0.0
    loss_pinn_epoch = 0.0
    
    for (point,u) in dataloader:
        point = point.clone()
        point = point.to(device)

        u = u.clone()
        u = u.to(device)
        
        pred = model(point)
        loss_data = loss_fn(pred,u)

        point_pinn = 60*torch.rand(5,3,requires_grad=True)-30        #random points at which derivates for Navier-Cauchy equation are obtained
        point_pinn = point_pinn.clone()
        point_pinn = point_pinn.to(device)

        is_inclusion = data.is_inclusion(point_pinn.detach().cpu().numpy()) #check which points are in inclusion and which in matrix
        lam = torch.where(is_inclusion == True, data.lambda_inclusion, data.lambda_matrix)
        mu = torch.where(is_inclusion == True, data.mu_inclusion, data.mu_matrix)

        lam = lam.clone()
        lam = lam.to(device)
        mu = mu.clone()
        mu = mu.to(device)

        u = model(point_pinn)
        res_x, res_y, res_z = residual(u, point_pinn, lam, mu)
        
        loss_pinn = loss_fn(res_x,torch.zeros_like(res_x)) + loss_fn(res_y,torch.zeros_like(res_y)) + loss_fn(res_z,torch.zeros_like(res_z))

        loss = loss_data + loss_pinn
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        
        loss_data_epoch += loss_data
        loss_pinn_epoch += loss_pinn

    loss_data_epoch /= len(dataloader)
    loss_pinn_epoch /= len(dataloader)
    #loss /= len(dataloader)

    return loss_data_epoch, loss_pinn_epoch
    #return loss
