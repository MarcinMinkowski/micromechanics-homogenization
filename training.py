import torch

def residual(ux, uy, uz, X, lam, mu):

    dux = torch.autograd.grad(outputs=ux,inputs=X,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
    duy = torch.autograd.grad(outputs=uy,inputs=X,grad_outputs=torch.ones_like(uy),retain_graph=True,create_graph=True)[0]
    duz = torch.autograd.grad(outputs=uz,inputs=X,grad_outputs=torch.ones_like(uz),retain_graph=True,create_graph=True)[0]

    dux_dx, dux_dy, dux_dz = dux[:,0], dux[:,1], dux[:,2]
    duy_dx, duy_dy, duy_dz = duy[:,0], duy[:,1], duy[:,2]
    duz_dx, duz_dy, duz_dz = duz[:,0], duz[:,1], duz[:,2]

    u_div = dux_dx + duy_dy + duz_dz

    du_div_d = torch.autograd.grad(outputs=u_div,inputs=X,grad_outputs=torch.ones_like(u_div),retain_graph=True)[0]

    dux_dxd = torch.autograd.grad(outputs=dux_dx,inputs=X,grad_outputs=torch.ones_like(dux_dx),retain_graph=True)[0]
    dux_dyd = torch.autograd.grad(outputs=dux_dy,inputs=X,grad_outputs=torch.ones_like(dux_dy),retain_graph=True)[0]
    dux_dzd = torch.autograd.grad(outputs=dux_dz,inputs=X,grad_outputs=torch.ones_like(dux_dz),retain_graph=True)[0]

    ux_laplacian = dux_dxd[:,0] + dux_dyd[:,1] + dux_dzd[:,2]
        
    duy_dxd = torch.autograd.grad(outputs=duy_dx,inputs=X,grad_outputs=torch.ones_like(duy_dx),retain_graph=True)[0]
    duy_dyd = torch.autograd.grad(outputs=duy_dy,inputs=X,grad_outputs=torch.ones_like(duy_dy),retain_graph=True)[0]
    duy_dzd = torch.autograd.grad(outputs=duy_dz,inputs=X,grad_outputs=torch.ones_like(duy_dz),retain_graph=True)[0]

    uy_laplacian = duy_dxd[:,0] + duy_dyd[:,1] + duy_dzd[:,2]

    duz_dxd = torch.autograd.grad(outputs=duz_dx,inputs=X,grad_outputs=torch.ones_like(duz_dx),retain_graph=True)[0]
    duz_dyd = torch.autograd.grad(outputs=duz_dy,inputs=X,grad_outputs=torch.ones_like(duz_dy),retain_graph=True)[0]
    duz_dzd = torch.autograd.grad(outputs=duz_dz,inputs=X,grad_outputs=torch.ones_like(duz_dz),retain_graph=True)[0]

    uz_laplacian = duz_dxd[:,0] + duz_dyd[:,1] + duz_dzd[:,2]

    return (lam+mu)*du_div_d[:,0]+mu*ux_laplacian, (lam+mu)*du_div_d[:,1]+mu*uy_laplacian, (lam+mu)*du_div_d[:,2]+mu*uz_laplacian

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

        point_pinn = 60*torch.rand(50,3,requires_grad=True)-30        #random points at which derivates for Navier-Cauchy equation are obtained
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
        
        res1_x, res1_y, res1_z = residual(u[:,0], u[:,1], u[:,2], point_pinn, lam, mu)
        res2_x, res2_y, res2_z = residual(u[:,3], u[:,4], u[:,5], point_pinn, lam, mu)
        res3_x, res3_y, res3_z = residual(u[:,6], u[:,7], u[:,8], point_pinn, lam, mu)
        res4_x, res4_y, res4_z = residual(u[:,9], u[:,10], u[:,11], point_pinn, lam, mu)
        res5_x, res5_y, res5_z = residual(u[:,12], u[:,13], u[:,14], point_pinn, lam, mu)
        res6_x, res6_y, res6_z = residual(u[:,15], u[:,16], u[:,17], point_pinn, lam, mu)
        
        loss_pinn = loss_fn(res1_x,torch.zeros_like(res1_x)) + loss_fn(res1_y,torch.zeros_like(res1_y)) + loss_fn(res1_z,torch.zeros_like(res1_z)) + \
                    loss_fn(res2_x,torch.zeros_like(res2_x)) + loss_fn(res2_y,torch.zeros_like(res2_y)) + loss_fn(res2_z,torch.zeros_like(res2_z)) + \
                    loss_fn(res3_x,torch.zeros_like(res3_x)) + loss_fn(res3_y,torch.zeros_like(res3_y)) + loss_fn(res3_z,torch.zeros_like(res3_z)) + \
                    loss_fn(res4_x,torch.zeros_like(res4_x)) + loss_fn(res4_y,torch.zeros_like(res4_y)) + loss_fn(res4_z,torch.zeros_like(res4_z)) + \
                    loss_fn(res5_x,torch.zeros_like(res5_x)) + loss_fn(res5_y,torch.zeros_like(res5_y)) + loss_fn(res5_z,torch.zeros_like(res5_z)) + \
                    loss_fn(res6_x,torch.zeros_like(res6_x)) + loss_fn(res6_y,torch.zeros_like(res6_y)) + loss_fn(res6_z,torch.zeros_like(res6_z))

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

def test_loop(model, dataloader, loss_fn, device):
    loss = 0.0
    with torch.no_grad():
        f = open("pred.dat","w")
        f.write("#u1x_true u1x_pred u1y_true u1y_pred u1z_true u1z_pred u2x_true u2x_pred u2y_true u2y_pred u2z_true u2z_pred u3x_true u3x_pred u3y_true u3y_pred u3z_true u3z_pred u4x_true u4x_pred u4y_true u4y_pred u4z_true u4z_pred u5x_true u5x_pred u5y_true u5y_pred u5z_true u5z_pred u6x_true u6x_pred u6y_true u6y_pred u6z_true u6z_pred\n")
        for (point, u) in dataloader:
            point = point.clone()
            point = point.to(device)

            u = u.clone()
            u = u.to(device)

            pred = model(point)
            loss += loss_fn(pred,u)

            for i, value in enumerate(pred[0]):
                f.write(str(u[0][i].item()) + " " + str(value.item()) + " ")
            f.write("\n")
        f.close()
    loss /= len(dataloader)
    return(loss)
