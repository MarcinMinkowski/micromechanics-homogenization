import torch

def residual(ux, uy, uz, X, lam, mu):

    dux = torch.autograd.grad(outputs=ux,inputs=X,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
    duy = torch.autograd.grad(outputs=uy,inputs=X,grad_outputs=torch.ones_like(uy),retain_graph=True,create_graph=True)[0]
    duz = torch.autograd.grad(outputs=uz,inputs=X,grad_outputs=torch.ones_like(uz),retain_graph=True,create_graph=True)[0]
    
    dux_dx, dux_dy, dux_dz = dux[:,0], dux[:,1], dux[:,2]
    duy_dx, duy_dy, duy_dz = duy[:,0], duy[:,1], duy[:,2]
    duz_dx, duz_dy, duz_dz = duz[:,0], duz[:,1], duz[:,2]

    u_div = dux_dx + duy_dy + duz_dz

    du_div_d = torch.autograd.grad(outputs=u_div,inputs=X,grad_outputs=torch.ones_like(u_div),retain_graph=True,create_graph=True)[0]

    dux_dxd = torch.autograd.grad(outputs=dux_dx,inputs=X,grad_outputs=torch.ones_like(dux_dx),retain_graph=True,create_graph=True)[0]
    dux_dyd = torch.autograd.grad(outputs=dux_dy,inputs=X,grad_outputs=torch.ones_like(dux_dy),retain_graph=True,create_graph=True)[0]
    dux_dzd = torch.autograd.grad(outputs=dux_dz,inputs=X,grad_outputs=torch.ones_like(dux_dz),retain_graph=True,create_graph=True)[0]

    ux_laplacian = dux_dxd[:,0] + dux_dyd[:,1] + dux_dzd[:,2]
        
    duy_dxd = torch.autograd.grad(outputs=duy_dx,inputs=X,grad_outputs=torch.ones_like(duy_dx),retain_graph=True,create_graph=True)[0]
    duy_dyd = torch.autograd.grad(outputs=duy_dy,inputs=X,grad_outputs=torch.ones_like(duy_dy),retain_graph=True,create_graph=True)[0]
    duy_dzd = torch.autograd.grad(outputs=duy_dz,inputs=X,grad_outputs=torch.ones_like(duy_dz),retain_graph=True,create_graph=True)[0]

    uy_laplacian = duy_dxd[:,0] + duy_dyd[:,1] + duy_dzd[:,2]

    duz_dxd = torch.autograd.grad(outputs=duz_dx,inputs=X,grad_outputs=torch.ones_like(duz_dx),retain_graph=True,create_graph=True)[0]
    duz_dyd = torch.autograd.grad(outputs=duz_dy,inputs=X,grad_outputs=torch.ones_like(duz_dy),retain_graph=True,create_graph=True)[0]
    duz_dzd = torch.autograd.grad(outputs=duz_dz,inputs=X,grad_outputs=torch.ones_like(duz_dz),retain_graph=True,create_graph=True)[0]

    uz_laplacian = duz_dxd[:,0] + duz_dyd[:,1] + duz_dzd[:,2]

    return (lam+mu)*du_div_d[:,0]+mu*ux_laplacian, (lam+mu)*du_div_d[:,1]+mu*uy_laplacian, (lam+mu)*du_div_d[:,2]+mu*uz_laplacian

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
            point_pinn = (60*torch.rand(50,3,device=device)-30).requires_grad_(True)        #random points at which derivates for Navier-Cauchy equation are obtained

            is_inclusion = data.is_inclusion(point_pinn.detach().cpu().numpy()) #check which points are in inclusion and which in matrix
            lam = torch.where(is_inclusion == True, data.lambda_inclusion, data.lambda_matrix)
            mu = torch.where(is_inclusion == True, data.mu_inclusion, data.mu_matrix)

            lam = lam.to(device)
            mu = mu.to(device)

            u_pinn = model(point_pinn)
            u_pinn = u_pinn*scale_tensor + mean_tensor

            u_pinn = u_pinn.view(50,6,3)

            res_x, res_y, res_z = residual(u_pinn[:,:,0], u_pinn[:,:,1], u_pinn[:,:,2], point_pinn, lam, mu)

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
