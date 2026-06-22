import torch

def train_loop(model, dataloader, loss_fn, optimizer, device):
    loss_data_epoch = 0.0
    loss_pinn_epoch = 0.0
    for (point,u,lambda_inclusion,mu_inclusion,lambda_matrix,mu_matrix,is_inclusion) in dataloader:
        point = point.clone()
        u = u.clone()
        #lambda_inclusion = lambda_inclusion.clone()
        #mu_inclusion = mu_inclusion.clone()
        #lambda_matrix = lambda_matrix.clone()
        #mu_matrix = mu_matrix.clone()
        #point.requires_grad_()

        point = point.to(device)
        u = u.to(device)
        #lambda_inclusion = lambda_inclusion.to(device)
        #mu_inclusion = mu_inclusion.to(device)
        #lambda_matrix = lambda_matrix.to(device)
        #mu_matrix = mu_matrix.to(device)

        pred = model(point)
        loss = loss_fn(pred,u)

        #du1x = torch.autograd.grad(outputs=pred[0][0],inputs=point,retain_graph=True,create_graph=True)[0][0]
        #du1y = torch.autograd.grad(outputs=pred[0][1],inputs=point,retain_graph=True,create_graph=True)[0][0]
        #du1z = torch.autograd.grad(outputs=pred[0][2],inputs=point,retain_graph=True,create_graph=True)[0][0]

        #du1x_dx, du1x_dy, du1x_dz = du1x[0], du1x[1], du1x[2]
        #du1y_dx, du1y_dy, du1y_dz = du1y[0], du1y[1], du1y[2]
        #du1z_dx, du1z_dy, du1z_dz = du1z[0], du1z[1], du1z[2]

        #u1_div = du1x_dx + du1y_dy + du1z_dz

        #du1_div_d = torch.autograd.grad(outputs=u1_div,inputs=point,retain_graph=True)[0][0]

        #du1x_dxd = torch.autograd.grad(outputs=du1x_dx,inputs=point,retain_graph=True)[0][0]
        #du1x_dyd = torch.autograd.grad(outputs=du1x_dy,inputs=point,retain_graph=True)[0][0]
        #du1x_dzd = torch.autograd.grad(outputs=du1x_dz,inputs=point,retain_graph=True)[0][0]

        #u1x_laplacian = du1x_dxd[0] + du1x_dyd[1] + du1x_dzd[2]
        
        #du1y_dxd = torch.autograd.grad(outputs=du1y_dx,inputs=point,retain_graph=True)[0][0]
        #du1y_dyd = torch.autograd.grad(outputs=du1y_dy,inputs=point,retain_graph=True)[0][0]
        #du1y_dzd = torch.autograd.grad(outputs=du1y_dz,inputs=point,retain_graph=True)[0][0]

        #u1y_laplacian = du1y_dxd[0] + du1y_dyd[1] + du1y_dzd[2]

        #du1z_dxd = torch.autograd.grad(outputs=du1z_dx,inputs=point,retain_graph=True)[0][0]
        #du1z_dyd = torch.autograd.grad(outputs=du1z_dy,inputs=point,retain_graph=True)[0][0]
        #du1z_dzd = torch.autograd.grad(outputs=du1z_dz,inputs=point,retain_graph=True)[0][0]

        #u1z_laplacian = du1z_dxd[0] + du1z_dyd[1] + du1z_dzd[2]

        #if is_inclusion:
            #lam = lambda_inclusion
            #mu = mu_inclusion
        #else:
            #lam = lambda_matrix
            #mu = mu_matrix

        #loss_pinn = loss_fn((lam+mu)*du1_div_d[0],-mu*u1x_laplacian) + loss_fn((lam+mu)*du1_div_d[1],-mu*u1y_laplacian) + loss_fn((lam+mu)*du1_div_d[2],-mu*u1z_laplacian)

        #loss = loss_data + loss_pinn
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        
        #loss_data_epoch += loss_data
        #loss_pinn_epoch += loss_pinn

    #loss_data_epoch /= len(dataloader)
    #loss_pinn_epoch /= len(dataloader)
    loss /= len(dataloader)

    #return loss_data_epoch, loss_pinn_epoch
    return loss
