import torch

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

        point_pinn = 60*torch.rand(3,requires_grad=True)-30        #random point at which derivates for Navier-Cauchy equation are obtained
        point_pinn = point_pinn.clone()
        point_pinn = point_pinn.to(device)
        
        pred_pinn = model(point_pinn.unsqueeze(0))

        du1x = torch.autograd.grad(outputs=pred_pinn[0][0],inputs=point_pinn,retain_graph=True,create_graph=True)[0]
        du1y = torch.autograd.grad(outputs=pred_pinn[0][1],inputs=point_pinn,retain_graph=True,create_graph=True)[0]
        du1z = torch.autograd.grad(outputs=pred_pinn[0][2],inputs=point_pinn,retain_graph=True,create_graph=True)[0]

        du1x_dx, du1x_dy, du1x_dz = du1x[0], du1x[1], du1x[2]
        du1y_dx, du1y_dy, du1y_dz = du1y[0], du1y[1], du1y[2]
        du1z_dx, du1z_dy, du1z_dz = du1z[0], du1z[1], du1z[2]

        u1_div = du1x_dx + du1y_dy + du1z_dz

        du1_div_d = torch.autograd.grad(outputs=u1_div,inputs=point_pinn,retain_graph=True)[0]

        du1x_dxd = torch.autograd.grad(outputs=du1x_dx,inputs=point_pinn,retain_graph=True)[0]
        du1x_dyd = torch.autograd.grad(outputs=du1x_dy,inputs=point_pinn,retain_graph=True)[0]
        du1x_dzd = torch.autograd.grad(outputs=du1x_dz,inputs=point_pinn,retain_graph=True)[0]

        u1x_laplacian = du1x_dxd[0] + du1x_dyd[1] + du1x_dzd[2]
        
        du1y_dxd = torch.autograd.grad(outputs=du1y_dx,inputs=point_pinn,retain_graph=True)[0]
        du1y_dyd = torch.autograd.grad(outputs=du1y_dy,inputs=point_pinn,retain_graph=True)[0]
        du1y_dzd = torch.autograd.grad(outputs=du1y_dz,inputs=point_pinn,retain_graph=True)[0]

        u1y_laplacian = du1y_dxd[0] + du1y_dyd[1] + du1y_dzd[2]

        du1z_dxd = torch.autograd.grad(outputs=du1z_dx,inputs=point_pinn,retain_graph=True)[0]
        du1z_dyd = torch.autograd.grad(outputs=du1z_dy,inputs=point_pinn,retain_graph=True)[0]
        du1z_dzd = torch.autograd.grad(outputs=du1z_dz,inputs=point_pinn,retain_graph=True)[0]

        u1z_laplacian = du1z_dxd[0] + du1z_dyd[1] + du1z_dzd[2]

        if data.is_inclusion(point_pinn.detach().numpy()):
            lam = data.lambda_inclusion
            mu = data.mu_inclusion
        else:
            lam = data.lambda_matrix
            mu = data.mu_matrix

        loss_pinn = loss_fn((lam+mu)*du1_div_d[0],-mu*u1x_laplacian) + loss_fn((lam+mu)*du1_div_d[1],-mu*u1y_laplacian) + loss_fn((lam+mu)*du1_div_d[2],-mu*u1z_laplacian)

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
