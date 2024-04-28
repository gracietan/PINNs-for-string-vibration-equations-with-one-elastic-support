#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as grad

torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Layer(nn.Module):
    def __init__(self, n_in, n_out, activation=None):
        super(Layer, self).__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

class WaveNet(nn.Module):
    def __init__(self, dim_in, dim_out=#dim_out, n_layer=#n_layer, n_node=#n_node, ub=None, lb=None, activation=#activation):
        super(WaveNet, self).__init__()
        self.net = nn.ModuleList()
        self.net.append(Layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(Layer(n_node, n_node, activation))
        self.net.append(Layer(n_node, dim_out, None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        for layer in self.net:
            x = layer(x)
        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

x_min =#minimum string length
x_es = #the needed location of the elastic support
x_max =#maximum string length
t_min =#minimum time
t_max =#maximum time
lb_1 = np.array([x_min, t_min])
ub_1 = np.array([x_es, t_max])
lb_2 = np.array([x_es, t_min])
ub_2 = np.array([x_max, t_max])


N_01 =#collocation points 
N_02 =#collocation points 
x_01 = np.linspace(x_min, x_es, N_01).reshape(-1, 1)
x_02 = np.linspace(x_es, x_max, N_02).reshape(-1, 1)
x_0 = np.linspace(x_min, x_max, N_01+ N_02).reshape(-1, 1)
t_01 = np.zeros((N_01, 1))
t_02 = np.zeros((N_02, 1))
t_0 =np.zeros((N_01+ N_02, 1))
t = np.linspace(t_min, t_max, N_01 + N_02).reshape(-1, 1)
xt_combinations = np.array(np.meshgrid(x_0.flatten(), t.flatten())).T.reshape(-1, 2)
xt_combinations = torch.tensor(xt_combinations, dtype=torch.float32).to(device)

x_es_t = np.ones((N_01 + N_02, 1)) * x_es
u_01 = #IC_1
u_02 = #IC_2
u_t_01 =#IC_t1
u_t_02 =#IC_t2

x_b_left = np.full((N_01, 1), x_min)
t_b_left = np.linspace(t_min, t_max, N_01).reshape(-1, 1)
xt_b_left = torch.tensor(np.hstack([x_b_left, t_b_left]),
dtype=torch.float32).to(device)

x_b_right = np.full((N_02, 1), x_max)
t_b_right = np.linspace(t_min, t_max, N_02).reshape(-1, 1)
xt_b_right = torch.tensor(np.hstack([x_b_right, t_b_right]),
dtype=torch.float32).to(device)

xt_01 = torch.tensor(np.hstack([x_01, t_01]), dtype=torch.float32).to(device)
xt_02 = torch.tensor(np.hstack([x_02, t_02]), dtype=torch.float32).to(device)
xt_0 = torch.tensor(np.hstack([x_0, t_0]), dtype=torch.float32).to(device)
xt_es_t = torch.tensor(np.hstack([x_es_t, t]), dtype=torch.float32).to(device)

u_01 = torch.tensor(u_01, dtype=torch.float32).to(device)
u_02 = torch.tensor(u_02, dtype=torch.float32).to(device)
u_t_01 = torch.tensor(u_t_01, dtype=torch.float32).to(device)
u_t_02 = torch.tensor(u_t_02, dtype=torch.float32).to(device)

xt_b_left = torch.tensor(np.hstack([x_b_left, t_b_left]),
dtype=torch.float32).to(device)
xt_b_right = torch.tensor(np.hstack([x_b_right, t_b_right]),
dtype=torch.float32).to(device)

class EsPINN:
    sigma =#sigma in the constraint condition
    a =#a in the equation
    def __init__(self):
        self.net = WaveNet(dim_in=#dim_in, n_layer=#n_layer, n_node=#n_node, ub=np.array([x_max, t_max]), 
                           lb=np.array([x_min, t_min])).to(device)
        self.net.apply(weights_init)
        self.optimizer = optim.Adam(self.net.parameters(), lr=#lr)

    def compute_loss(self, xt, xt_1, xt_2, u, u_t, v, v_t, xt_e_t, xt_comb, xt_b_left, xt_b_right):
        xt.requires_grad = True
        xt_1.requires_grad = True
        xt_2.requires_grad = True
        xt_e_t.requires_grad = True
        xt_comb.requires_grad = True
        mask_1 = (xt_comb[:, 0:1] <= x_es).type(torch.float32)
        mask_2 = (xt_comb[:, 0:1] >= x_es).type(torch.float32)

        net_output_u = self.net(xt_1)
        net_output_v = self.net(xt_2)
        net_output_es = self.net(xt_e_t)
        net_output_comb = self.net(xt_comb)
        net_output_b_left = self.net(xt_b_left)
        net_output_b_right = self.net(xt_b_right)
        
        u_pred_es = net_output_es[:, 0:1]
        v_pred_es = net_output_es[:, 1:2]

        u_pred = net_output_u[:, 0:1]
        v_pred = net_output_v[:, 1:2]

        u_pred_comb = net_output_comb[:, 0:1]
        v_pred_comb = net_output_comb[:, 1:2]

        u_xt_comb = grad.grad(u_pred_comb.sum(), xt_comb, create_graph=True)[0]
        u_xx_comb = grad.grad(u_xt_comb[0:, 0:1].sum(), xt_comb, create_graph=True)[0][0:, 0:1]
        u_tt_comb = grad.grad(u_xt_comb[0:, 1:2].sum(), xt_comb, create_graph=True)[0][0:, 1:2]

        v_xt_comb = grad.grad(v_pred_comb.sum(), xt_comb, create_graph=True)[0]
        v_xx_comb = grad.grad(v_xt_comb[0:, 0:1].sum(), xt_comb, create_graph=True)[0][0:, 0:1]
        v_tt_comb = grad.grad(v_xt_comb[0:, 1:2].sum(), xt_comb, create_graph=True)[0][0:, 1:2]


        u_t_pred = grad.grad(u_pred.sum(), xt_1, create_graph=True)[0][:, 1:2]
        v_t_pred = grad.grad(v_pred.sum(), xt_2, create_graph=True)[0][:, 1:2]

        loss_u = (lambda_u)*torch.mean( (u_01 - u_pred) ** 2)
        loss_u_t = (lambda_ut)*torch.mean( (u_t - u_t_pred) ** 2)

        loss_v = (lambda_v)*torch.mean((u_02 - v_pred) ** 2)
        loss_v_t =(lambda_vt)* torch.mean( (v_t - v_t_pred) ** 2)

        loss_es = (lambda_es)*torch.mean((u_pred_es - v_pred_es)**2)

        u_x_pred = grad.grad(u_pred_es.sum(), xt_e_t, create_graph=True)[0][:, 0:1]
        v_x_pred = grad.grad(v_pred_es.sum(), xt_e_t, create_graph=True)[0][:, 0:1]

        loss_bindingcondition= (lambda_bindingcondition)*torch.mean((v_x_pred - u_x_pred - self.sigma*u_pred_es)**2)

        pde_residuals_1 = u_tt_comb - (self.a ** 2) * u_xx_comb  
        pde_residuals_masked_1 = pde_residuals_1 * mask_1

        pde_residuals_2 = v_tt_comb - (self.a**2) * v_xx_comb
        pde_residuals_masked_2 = pde_residuals_2*mask_2

        loss_pde = (lambda_pde1)*torch.mean(pde_residuals_masked_1**2) + (lambda_pde2)*torch.mean(pde_residuals_masked_2**2)
        
        loss_boundary_left =(lambda_bl)*torch.mean(net_output_b_left ** 2)
        loss_boundary_right =(lambda_br)*torch.mean(net_output_b_right ** 2)
        
        return loss_u + loss_u_t + loss_v + loss_v_t + loss_es + loss_bindingcondition + loss_pde+loss_boundary_left + loss_boundary_right


    def closure(self):
        self.optimizer.zero_grad()
        loss = self.compute_loss(xt_0, xt_01, xt_02, u_01, u_t_01, u_02, u_t_02, xt_es_t, xt_combinations, xt_b_left, xt_b_right)
        loss.backward()
        return loss

    def train(self, epochs):
        losses = []
        for epoch in range(epochs):
            loss = self.optimizer.step(self.closure)
            losses.append(loss.item())
            if (epoch) % #number == 0:
                print(f'Epoch {epoch}: Loss={loss.item()}')
            if epoch ==#epoch:
                print(f'Epoch {epoch}: Loss={loss.item()}')
        
        plt.plot(range(1, epochs + 1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

        print('Training complete')

def plot_u_vs_x(t_values, es_pinn, n_points=1000):
    x = np.linspace(x_min, x_max, n_points)
    plt.figure(figsize=(12, 6))
    
    for t in t_values:
        t_array = np.full((n_points, 1), t) 
        xt = np.hstack((x.reshape(-1, 1), t_array)) 
        xt_tensor = torch.tensor(xt, dtype=torch.float32).to(device)
        
        net_output = es_pinn.net(xt_tensor).detach().cpu().numpy()
        u = net_output[:, 0] 
        v = net_output[:, 1] 

        u_masked = np.where(x <= x_es, u, np.nan) 
        v_masked = np.where(x >= x_es, v, np.nan) 

        plt.plot(x, u_masked, label=f'u at t = {t}')
        plt.plot(x, v_masked, label=f'v at t = {t}')

    plt.xlabel('x')
    plt.ylabel('Values')
    plt.title('U profile at different times')
    plt.legend()
    plt.tight_layout()
    plt.show()

model = EsPINN()
model.train(epochs=#epochs)
t_values = [#the needed interval of t]
plot_u_vs_x(t_values, model)

