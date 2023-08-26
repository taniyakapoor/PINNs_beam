import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Define the exact solution (Phi1)
def exact_solution_u(x, t):
    return (pi/2*torch.cos(x) + (x-pi/2))*torch.cos(t)

def initial_condition_u(x):
    return (pi/2*torch.cos(x) + (x-pi/2))

def initial_condition_u_t(x):
    return 0.0*pi*torch.cos(x) 

# Define the exact solution(w1)
def exact_solution_p(x, t):
    return pi/2*torch.sin(x)*torch.cos(t)

def initial_condition_p(x):
    return pi/2*torch.sin(x)

def initial_condition_p_t(x):
    return 0*torch.sin(x) 

# Define the exact solution (Phi1)
def exact_solution_u1(x, t):
    return (2/pi)*(pi/2*torch.cos(x) + (x-pi/2))*torch.cos(t)

def initial_condition_u1(x):
    return (2/pi)*(pi/2*torch.cos(x) + (x-pi/2))

def initial_condition_u_t1(x):
    return 0.0*pi*torch.cos(x) 

# Define the exact solution(w1)
def exact_solution_p1(x, t):
    return torch.sin(x)*torch.cos(t)

def initial_condition_p1(x):
    return torch.sin(x)

def initial_condition_p_t1(x):
    return 0*torch.sin(x) 

# assigning number of points
initial_pts = 2000
left_boundary_pts = 2000
right_boundary_pts = 2000
residual_pts = 10000
# Type of optimizer (ADAM or LBFGS)
opt_type = "LBFGS"
manualSeed = 1

#np.random.seed(manualSeed)
#random.seed(manualSeed)
torch.manual_seed(manualSeed)

x_init = pi*torch.rand((initial_pts,1)) # initial pts
t_init = 0*x_init
init =  torch.cat([x_init, t_init],1)
u_init = initial_condition_u(init[:,0]).reshape(-1, 1)
p_init = initial_condition_p(init[:,0]).reshape(-1, 1)
u1_init = initial_condition_u1(init[:,0]).reshape(-1, 1)
p1_init = initial_condition_p1(init[:,0]).reshape(-1, 1)
w_init = torch.cat([u_init, p_init, u1_init, p1_init],1)

u_t_init = initial_condition_u_t(init[:,0]).reshape(-1, 1)
p_t_init = initial_condition_p_t(init[:,0]).reshape(-1, 1)
u1_t_init = initial_condition_u_t1(init[:,0]).reshape(-1, 1)
p1_t_init = initial_condition_p_t1(init[:,0]).reshape(-1, 1)
w_t_init = torch.cat([u_t_init, p_t_init, u1_t_init, p1_t_init],1)



xb_left = torch.zeros((left_boundary_pts, 1)) # left spatial boundary
tb_left = torch.rand((left_boundary_pts, 1)) # 
b_left = torch.cat([xb_left, tb_left ],1)
u_b_l = exact_solution_u(xb_left, tb_left)
p_b_l = exact_solution_p(xb_left, tb_left)
u1_b_l = exact_solution_u1(xb_left, tb_left)
p1_b_l = exact_solution_p1(xb_left, tb_left)
w_b_l = torch.cat([u_b_l, p_b_l, u1_b_l, p1_b_l],1)



xb_right = pi*torch.ones((right_boundary_pts, 1)) # right spatial boundary
tb_right = torch.rand((right_boundary_pts, 1)) # right boundary pts
b_right = torch.cat([xb_right, tb_right ],1)
u_b_r = exact_solution_u(xb_right, tb_right)
p_b_r = exact_solution_p(xb_right, tb_right)
u1_b_r = exact_solution_u1(xb_right, tb_right)
p1_b_r = exact_solution_p1(xb_right, tb_right)
w_b_r = torch.cat([u_b_r, p_b_r, u1_b_r, p1_b_r],1)

x_interior = pi*torch.rand((residual_pts, 1))
t_interior = torch.rand((residual_pts, 1))
interior = torch.cat([x_interior, t_interior],1)





training_set = DataLoader(torch.utils.data.TensorDataset(init, w_init, w_t_init, b_left,  b_right), batch_size=2000, shuffle=False)

class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer 
        self.neurons = neurons
        # Number of hidden layers 
        self.n_hidden_layers = n_hidden_layers
        # Activation function 
        self.activation = nn.Tanh()
        
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network 
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)
# Model definition
my_network = NeuralNet(input_dimension = init.shape[1], output_dimension = w_init.shape[1], n_hidden_layers=4, neurons=20)

def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            #torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)
    model.apply(init_weights)

# Random Seed for weight initialization
retrain = 128
# Xavier weight initialization
init_xavier(my_network, retrain)
#print(my_network(init))

if opt_type == "ADAM":
    optimizer_ = optim.Adam(my_network.parameters(), lr=0.001)
elif opt_type == "LBFGS":
    optimizer_ = optim.LBFGS(my_network.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)
else:
    raise ValueError("Optimizer not recognized")
    
def fit(model, training_set, interior, num_epochs, optimizer, p, verbose=True):
    history = list()
    
    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = list([0])
        
        # Loop over batches
        for j, (initial, w_initial, w_initial_t, bd_left,  bd_right) in enumerate(training_set):
            
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                                
                bd_left.requires_grad = True
                bd_right.requires_grad = True
                # for initial
                initial.requires_grad = True
                w_initial_pred_ = model(initial)
                u_initial_pred_ = w_initial_pred_[:,0].reshape(-1,1)
                p_initial_pred_ = w_initial_pred_[:,1].reshape(-1,1)
                u1_initial_pred_ = w_initial_pred_[:,2].reshape(-1,1)
                p1_initial_pred_ = w_initial_pred_[:,3].reshape(-1,1)
                
                

                
                
                # with derivative
                inpu = torch.ones(initial_pts, 1 )
                
                grad_u_ini = torch.autograd.grad(u_initial_pred_, initial, grad_outputs=inpu, create_graph=True, allow_unused=True)[0]
                
                u_initial_t = grad_u_ini[:, 1]
                
                
                grad_p_ini = torch.autograd.grad(p_initial_pred_, initial, grad_outputs=inpu, create_graph=True)[0]
                
                p_initial_t = grad_p_ini[:, 1]
                
                grad_u1_ini = torch.autograd.grad(u1_initial_pred_, initial, grad_outputs=inpu, create_graph=True, allow_unused=True)[0]
                
                u1_initial_t = grad_u1_ini[:, 1]
                
                
                grad_p1_ini = torch.autograd.grad(p1_initial_pred_, initial, grad_outputs=inpu, create_graph=True)[0]
                
                p1_initial_t = grad_p1_ini[:, 1]
                
                
                

                
                

                # for left boundary
                w_bd_left_pred_ = model(bd_left)
                u_bd_left_pred_ = w_bd_left_pred_[:,0].reshape(-1,1)
                p_bd_left_pred_ = w_bd_left_pred_[:,1].reshape(-1,1)
                
                u1_bd_left_pred_ = w_bd_left_pred_[:,2].reshape(-1,1)
                p1_bd_left_pred_ = w_bd_left_pred_[:,3].reshape(-1,1)
                
                # for right boundary
                w_bd_right_pred_ = model(bd_right)
                u_bd_right_pred_ = w_bd_right_pred_[:,0].reshape(-1,1)
                p_bd_right_pred_ = w_bd_right_pred_[:,1].reshape(-1,1)
                
                u1_bd_right_pred_ = w_bd_right_pred_[:,2].reshape(-1,1)
                p1_bd_right_pred_ = w_bd_right_pred_[:,3].reshape(-1,1)
                


                inputs2 = torch.ones(left_boundary_pts, 1)
                inputs3 = torch.ones(right_boundary_pts, 1)
                grad_u_b_l = torch.autograd.grad(u_bd_left_pred_, bd_left, grad_outputs=inputs2, create_graph=True)[0]
                grad_u_b_r = torch.autograd.grad(u_bd_right_pred_, bd_right, grad_outputs=inputs3, create_graph=True)[0]
                u_b_l_x = grad_u_b_l[:, 0]
                u_b_r_x = grad_u_b_r[:, 0]
                
                u_b_l_xx = torch.autograd.grad(u_b_l_x, bd_left, grad_outputs=torch.ones(bd_left.shape[0]), create_graph=True)[0]
                u_bd_left_xx = u_b_l_xx[:, 0].reshape(-1,1)

                u_b_r_xx = torch.autograd.grad(u_b_r_x, bd_right, grad_outputs=torch.ones(bd_right.shape[0]), create_graph=True)[0]
                u_bd_right_xx = u_b_r_xx[:, 0].reshape(-1,1)
                
                inputs4 = torch.ones(left_boundary_pts, 1)
                inputs5 = torch.ones(right_boundary_pts, 1)
                grad_v_b_l = torch.autograd.grad(p_bd_left_pred_, bd_left, grad_outputs=inputs4, create_graph=True)[0]
                grad_v_b_r = torch.autograd.grad(p_bd_right_pred_, bd_right, grad_outputs=inputs5, create_graph=True)[0]
                v_b_l_x = grad_v_b_l[:, 0]
                v_b_r_x = grad_v_b_r[:, 0]
                
                v_b_l_xx = torch.autograd.grad(v_b_l_x, bd_left, grad_outputs=torch.ones(bd_left.shape[0]), create_graph=True)[0]
                v_bd_left_xx = v_b_l_xx[:, 0].reshape(-1,1)

                v_b_r_xx = torch.autograd.grad(v_b_r_x, bd_right, grad_outputs=torch.ones(bd_right.shape[0]), create_graph=True)[0]
                v_bd_right_xx = v_b_r_xx[:, 0].reshape(-1,1)
                

                
                
                
               
                
                
                # residual calculation
                interior.requires_grad = True
                w_hat = model(interior)
                u_hat = w_hat[:,0].reshape(-1,1)
                p_hat = w_hat[:,1].reshape(-1,1)
                u1_hat = w_hat[:,2].reshape(-1,1)
                p1_hat = w_hat[:,3].reshape(-1,1)
                
                inputs = torch.ones(residual_pts, 1 )
                inputs2 = torch.ones(residual_pts, 1)
                grad_u_hat = torch.autograd.grad(u_hat.reshape(-1,1), interior, grad_outputs=inputs, create_graph=True)[0]
                
                u_x = grad_u_hat[:, 0].reshape(-1,1)
                
                grad_u_hat_xx = torch.autograd.grad(u_x, interior, grad_outputs=inputs, create_graph=True)[0]
                
                u_xx = grad_u_hat_xx[:, 0].reshape(-1,1)
                
                grad_u_hat_xxx = torch.autograd.grad(u_xx, interior, grad_outputs=inputs, create_graph=True)[0]
                
                u_xxx = grad_u_hat_xxx[:, 0].reshape(-1,1)
                
                grad_u_hat_xxxx = torch.autograd.grad(u_xxx, interior, grad_outputs=inputs, create_graph=True)[0]
                
                u_xxxx = grad_u_hat_xxxx[:, 0].reshape(-1,1)
                
                
                
                
                
                grad_p_hat = torch.autograd.grad(p_hat, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p_x = grad_p_hat[:, 0].reshape(-1,1)
                
                grad_p_hat_xx = torch.autograd.grad(p_x, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p_xx = grad_p_hat_xx[:, 0].reshape(-1,1)
                
                grad_p_hat_xxx = torch.autograd.grad(p_xx, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p_xxx = grad_p_hat_xxx[:, 0].reshape(-1,1)
                
                grad_p_hat_xxxx = torch.autograd.grad(p_xxx, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p_xxxx = grad_p_hat_xxxx[:, 0].reshape(-1,1)
                
                
                grad_u1_hat = torch.autograd.grad(u1_hat.reshape(-1,1), interior, grad_outputs=inputs, create_graph=True)[0]
                
                u1_x = grad_u1_hat[:, 0].reshape(-1,1)
                
                grad_u1_hat_xx = torch.autograd.grad(u1_x, interior, grad_outputs=inputs, create_graph=True)[0]
                
                u1_xx = grad_u1_hat_xx[:, 0].reshape(-1,1)
                
                
                
                
                grad_p1_hat = torch.autograd.grad(p1_hat, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p1_x = grad_p1_hat[:, 0].reshape(-1,1)
                
                grad_p1_hat_xx = torch.autograd.grad(p1_x, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p1_xx = grad_p1_hat_xx[:, 0].reshape(-1,1)
                
                
                

                
                
                
                
                #grad_grad_u_x = torch.autograd.grad(u_x, interior, grad_outputs=torch.ones(interior.shape[0]), create_graph=True)[0]
                #u_xx = grad_grad_u_x[:, 0]
                u_t = grad_u_hat[:, 1].reshape(-1,1)
                
                grad_u_hat_tt = torch.autograd.grad(u_t, interior, grad_outputs=inputs, create_graph=True)[0]
                
                u_tt = grad_u_hat_tt[:, 1].reshape(-1,1)
                
                p_t = grad_p_hat[:,1].reshape(-1,1)
                grad_p_hat_tt = torch.autograd.grad(p_t, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p_tt = grad_p_hat_tt[:, 1].reshape(-1,1)
                
                
                u1_t = grad_u1_hat[:, 1].reshape(-1,1)
                
                grad_u1_hat_tt = torch.autograd.grad(u1_t, interior, grad_outputs=inputs, create_graph=True)[0]
                
                u1_tt = grad_u1_hat_tt[:, 1].reshape(-1,1)
                
                p1_t = grad_p1_hat[:,1].reshape(-1,1)
                grad_p1_hat_tt = torch.autograd.grad(p1_t, interior, grad_outputs=inputs, create_graph=True)[0]
                
                p1_tt = grad_p1_hat_tt[:, 1].reshape(-1,1)
                
                
                
                
                
                # Item 1. below
                loss1 = torch.mean((u_initial_t.reshape(-1, ) - w_initial_t[:,0].reshape(-1, ))**p)+torch.mean((u_initial_pred_.reshape(-1, ) - w_initial[:,0].reshape(-1, ))**p) + torch.mean((u_x.reshape(-1, ) - p_xx.reshape(-1, ) + p_tt.reshape(-1, ) + p_hat.reshape(-1, )-p1_hat.reshape(-1, ) - (1-torch.sin(interior[:, 0]))*torch.cos(interior[:, 1]).reshape(-1, ))**p) + torch.mean((u_bd_left_pred_.reshape(-1,)- u_b_l.reshape(-1,))**p) + torch.mean((u_bd_right_pred_.reshape(-1,)- u_b_r.reshape(-1,))**p)
                
                
                loss2 = torch.mean((p_initial_pred_.reshape(-1, ) - w_initial[:,1].reshape(-1, ))**p)+ torch.mean((p_initial_t.reshape(-1, ) - w_initial_t[:,1].reshape(-1, ))**p) + torch.mean((u_xx.reshape(-1, )  + p_x.reshape(-1, ) - u_tt.reshape(-1, ) - u_hat.reshape(-1, ))**p)+torch.mean((p_bd_left_pred_.reshape(-1,)- p_b_l.reshape(-1,))**p) + torch.mean((p_bd_right_pred_.reshape(-1,)- p_b_r.reshape(-1,))**p)

                loss3 = torch.mean((u1_initial_t.reshape(-1, ) - w_initial_t[:,2].reshape(-1, ))**p)+torch.mean((u1_initial_pred_.reshape(-1, ) - w_initial[:,2].reshape(-1, ))**p) + torch.mean((u1_x.reshape(-1, ) - p1_xx.reshape(-1, ) + p1_tt.reshape(-1, ) + p1_hat.reshape(-1, ) - p_hat.reshape(-1, ).reshape(-1, )-(2/pi)*torch.cos(interior[:, 1])+pi/2*torch.sin(interior[:,0])*torch.cos(interior[:, 1]))**p) + torch.mean((u1_bd_left_pred_.reshape(-1,))**p) + torch.mean((u1_bd_right_pred_.reshape(-1,))**p)
                
                loss4 = torch.mean((p1_initial_pred_.reshape(-1, ) - w_initial[:,3].reshape(-1, ))**p)+ torch.mean((p1_initial_t.reshape(-1, ) - w_initial_t[:,3].reshape(-1, ))**p) + torch.mean((u1_xx.reshape(-1, )  + p1_x.reshape(-1, ) - u1_tt.reshape(-1, ) - u1_hat.reshape(-1, ))**p)+torch.mean((p1_bd_left_pred_.reshape(-1,))**p) + torch.mean((p1_bd_right_pred_.reshape(-1,))**p)
                


                loss = loss1 + loss2 + loss3 + loss4
                #loss = torch.max(torch.abs((u_initial_pred_.reshape(-1, ) - u_initial.reshape(-1, )))) + torch.max(torch.abs((u_t.reshape(-1, ) - u_xx.reshape(-1, ))))+torch.max(torch.abs((u_bd_left_pred_.reshape(-1,)))) + torch.max(torch.abs((u_bd_right_pred_.reshape(-1,))))
 
                
                # Item 2. below
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item()
                return loss
            
            # Item 3. below
            optimizer.step(closure=closure)
            
        print('Loss: ', (running_loss[0] / len(training_set)))
        history.append(running_loss[0])

    return history
n_epochs = 15000
#history = fit(my_network, training_set, interior, n_epochs, optimizer_, p=2, verbose=True )
# saving and loading Model
FILE = "inverse.pth"
#torch.save(my_network, FILE)

# uncomment below when you need to test for different points
my_network = torch.load(FILE)
my_network.eval()

model = my_network

# for inverse problem

x1 = 0.2*torch.ones(100000).reshape(-1, 1)
t1 = torch.rand(100000).reshape(-1, 1)
test1 = torch.cat([x1, t1], 1)

w1 = my_network(test1)
print(w1.shape)
u1 = w1[:,0].reshape(-1,1)
p1 = w1[:,1].reshape(-1,1)
q1 = w1[:,2].reshape(-1,1)
r1 = w1[:,3].reshape(-1,1)

data_u1 = torch.cat([t1, x1, u1],1)
np.savetxt('timo_data_u1_inverse.csv' , [p for p in data_u1.detach().numpy()],  delimiter=',' , fmt='%s')
data_p1 = torch.cat([t1, x1, p1],1)
np.savetxt('timo_data_p1_inverse.csv' , [p for p in data_p1.detach().numpy()],  delimiter=',' , fmt='%s')
data_q1 = torch.cat([t1, x1, q1],1)
np.savetxt('timo_data_q1_inverse.csv' , [p for p in data_q1.detach().numpy()],  delimiter=',' , fmt='%s')
data_r1 = torch.cat([t1, x1, r1],1)
np.savetxt('timo_data_r1_inverse.csv' , [p for p in data_r1.detach().numpy()],  delimiter=',' , fmt='%s')


x2 = 0.4*torch.ones(100000).reshape(-1,1)
t2 = torch.rand(100000).reshape(-1,1)
test2 = torch.cat([x2, t2],1)

w2 = my_network(test2)
u2 = w2[:,0].reshape(-1,1)
p2 = w2[:,1].reshape(-1,1)
q2 = w2[:,2].reshape(-1,1)
r2 = w2[:,3].reshape(-1,1)

data_u2 = torch.cat([t2, x2, u2],1)
np.savetxt('timo_data_u2_inverse.csv' , [p for p in data_u2.detach().numpy()],  delimiter=',' , fmt='%s')
data_p2 = torch.cat([t2, x2, p2],1)
np.savetxt('timo_data_p2_inverse.csv' , [p for p in data_p2.detach().numpy()],  delimiter=',' , fmt='%s')
data_q2 = torch.cat([t2, x2, q2],1)
np.savetxt('timo_data_q2_inverse.csv' , [p for p in data_q2.detach().numpy()],  delimiter=',' , fmt='%s')
data_r2 = torch.cat([t2, x2, r2],1)
np.savetxt('timo_data_r2_inverse.csv' , [p for p in data_r2.detach().numpy()],  delimiter=',' , fmt='%s')



x3 = 0.6*torch.ones(100000).reshape(-1,1)
t3 = torch.rand(100000).reshape(-1,1)
test3 = torch.cat([x3, t3],1)

w3 = my_network(test3)
u3 = w3[:,0].reshape(-1,1)
p3 = w3[:,1].reshape(-1,1)
q3 = w3[:,2].reshape(-1,1)
r3 = w3[:,3].reshape(-1,1)

data_u3 = torch.cat([t3, x3, u3],1)
np.savetxt('timo_data_u3_inverse.csv' , [p for p in data_u3.detach().numpy()],  delimiter=',' , fmt='%s')
data_p3 = torch.cat([t3, x3, p3],1)
np.savetxt('timo_data_p3_inverse.csv' , [p for p in data_p3.detach().numpy()],  delimiter=',' , fmt='%s')
data_q3 = torch.cat([t3, x3, q3],1)
np.savetxt('timo_data_q3_inverse.csv' , [p for p in data_q3.detach().numpy()],  delimiter=',' , fmt='%s')
data_r3 = torch.cat([t3, x3, r3],1)
np.savetxt('timo_data_r3_inverse.csv' , [p for p in data_r3.detach().numpy()],  delimiter=',' , fmt='%s')

x4 = 0.8*torch.ones(100000).reshape(-1,1)
t4 = torch.rand(100000).reshape(-1,1)
test4 = torch.cat([x4, t4],1)

w4= my_network(test4)
u4= w4[:,0].reshape(-1,1)
p4= w4[:,1].reshape(-1,1)
q4 = w4[:,2].reshape(-1,1)
r4 = w4[:,3].reshape(-1,1)



data_u4 = torch.cat([t4, x4, u4],1)
np.savetxt('timo_data_u4_inverse.csv' , [p for p in data_u4.detach().numpy()],  delimiter=',' , fmt='%s')
data_p4 = torch.cat([t4, x4, p4],1)
np.savetxt('timo_data_p4_inverse.csv' , [p for p in data_p4.detach().numpy()],  delimiter=',' , fmt='%s')
data_q4 = torch.cat([t4, x4, q4],1)
np.savetxt('timo_data_q4_inverse.csv' , [p for p in data_q4.detach().numpy()],  delimiter=',' , fmt='%s')
data_r4 = torch.cat([t4, x4, r4],1)
np.savetxt('timo_data_r4_inverse.csv' , [p for p in data_r4.detach().numpy()],  delimiter=',' , fmt='%s')

x5 = torch.ones(100000).reshape(-1,1)
t5 =torch.rand(100000).reshape(-1,1)
test5 = torch.cat([x5, t5],1)

w5= my_network(test5)
u5= w5[:,0].reshape(-1,1)
p5= w5[:,1].reshape(-1,1)
q5 = w5[:,2].reshape(-1,1)
r5 = w5[:,3].reshape(-1,1)


data_u5 = torch.cat([t5, x5, u5],1)
np.savetxt('timo_data_u5_inverse.csv' , [p for p in data_u5.detach().numpy()],  delimiter=',' , fmt='%s')
data_p5 = torch.cat([t5, x5, p5],1)
np.savetxt('timo_data_p5_inverse.csv' , [p for p in data_p5.detach().numpy()],  delimiter=',' , fmt='%s')
data_q5 = torch.cat([t5, x5, q5],1)
np.savetxt('timo_data_q5_inverse.csv' , [p for p in data_q5.detach().numpy()],  delimiter=',' , fmt='%s')
data_r5 = torch.cat([t5, x5, r5],1)
np.savetxt('timo_data_r5_inverse.csv' , [p for p in data_r5.detach().numpy()],  delimiter=',' , fmt='%s')

x6 = 1.2*torch.ones(100000).reshape(-1,1)
t6 =torch.rand(100000).reshape(-1,1)
test6 = torch.cat([x6, t6],1)

w6= my_network(test6)
u6= w6[:,0].reshape(-1,1)
p6= w6[:,1].reshape(-1,1)
q6 = w6[:,2].reshape(-1,1)
r6 = w6[:,3].reshape(-1,1)

data_u6 = torch.cat([t6, x6, u6],1)
np.savetxt('timo_data_u6_inverse.csv' , [p for p in data_u6.detach().numpy()],  delimiter=',' , fmt='%s')
data_p6 = torch.cat([t6, x6, p6],1)
np.savetxt('timo_data_p6_inverse.csv' , [p for p in data_p6.detach().numpy()],  delimiter=',' , fmt='%s')
data_q6 = torch.cat([t6, x6, q6],1)
np.savetxt('timo_data_q6_inverse.csv' , [p for p in data_q6.detach().numpy()],  delimiter=',' , fmt='%s')
data_r6 = torch.cat([t6, x6, r6],1)
np.savetxt('timo_data_r6_inverse.csv' , [p for p in data_r6.detach().numpy()],  delimiter=',' , fmt='%s')

x7 = 1.6*torch.ones(100000).reshape(-1,1)
t7 =torch.rand(100000).reshape(-1,1)
test7 = torch.cat([x7, t7],1)

w7= my_network(test7)
u7= w7[:,0].reshape(-1,1)
p7= w7[:,1].reshape(-1,1)
q7 = w7[:,2].reshape(-1,1)
r7 = w7[:,3].reshape(-1,1)

data_u7 = torch.cat([t7, x7, u7],1)
np.savetxt('timo_data_u7_inverse.csv' , [p for p in data_u7.detach().numpy()],  delimiter=',' , fmt='%s')
data_p7 = torch.cat([t7, x7, p7],1)
np.savetxt('timo_data_p7_inverse.csv' , [p for p in data_p7.detach().numpy()],  delimiter=',' , fmt='%s')
data_q7 = torch.cat([t7, x7, q7],1)
np.savetxt('timo_data_q7_inverse.csv' , [p for p in data_q7.detach().numpy()],  delimiter=',' , fmt='%s')
data_r7 = torch.cat([t7, x7, r7],1)
np.savetxt('timo_data_r7_inverse.csv' , [p for p in data_r7.detach().numpy()],  delimiter=',' , fmt='%s')


x8 = 1.8*torch.ones(100000).reshape(-1,1)
t8 =torch.rand(100000).reshape(-1,1)
test8 = torch.cat([x8, t8],1)

w8= my_network(test8)
u8= w8[:,0].reshape(-1,1)
p8= w8[:,1].reshape(-1,1)
q8 = w8[:,2].reshape(-1,1)
r8 = w8[:,3].reshape(-1,1)

data_u8 = torch.cat([t8, x8, u8],1)
np.savetxt('timo_data_u8_inverse.csv' , [p for p in data_u8.detach().numpy()],  delimiter=',' , fmt='%s')
data_p8 = torch.cat([t8, x8, p8],1)
np.savetxt('timo_data_p8_inverse.csv' , [p for p in data_p8.detach().numpy()],  delimiter=',' , fmt='%s')
data_q8 = torch.cat([t8, x8, q8],1)
np.savetxt('timo_data_q8_inverse.csv' , [p for p in data_q8.detach().numpy()],  delimiter=',' , fmt='%s')
data_r8 = torch.cat([t8, x8, r8],1)
np.savetxt('timo_data_r8_inverse.csv' , [p for p in data_r8.detach().numpy()],  delimiter=',' , fmt='%s')

x9 = 2*torch.ones(100000).reshape(-1,1)
t9 =torch.rand(100000).reshape(-1,1)
test9 = torch.cat([x9, t9],1)

w9= my_network(test9)
u9= w9[:,0].reshape(-1,1)
p9= w9[:,1].reshape(-1,1)
q9 = w9[:,2].reshape(-1,1)
r9 = w9[:,3].reshape(-1,1)

data_u9 = torch.cat([t9, x9, u9],1)
np.savetxt('timo_data_u9_inverse.csv' , [p for p in data_u9.detach().numpy()],  delimiter=',' , fmt='%s')
data_p9 = torch.cat([t9, x9, p9],1)
np.savetxt('timo_data_p9_inverse.csv' , [p for p in data_p9.detach().numpy()],  delimiter=',' , fmt='%s')
data_q9 = torch.cat([t9, x9, q9],1)
np.savetxt('timo_data_q9_inverse.csv' , [p for p in data_q9.detach().numpy()],  delimiter=',' , fmt='%s')
data_r9 = torch.cat([t9, x9, r9],1)
np.savetxt('timo_data_r9_inverse.csv' , [p for p in data_r9.detach().numpy()],  delimiter=',' , fmt='%s')

x10 = 2.2*torch.ones(100000).reshape(-1,1)
t10 =torch.rand(100000).reshape(-1,1)
test10 = torch.cat([x10, t10],1)

w10= my_network(test10)
u10= w10[:,0].reshape(-1,1)
p10= w10[:,1].reshape(-1,1)
q10 = w10[:,2].reshape(-1,1)
r10 = w10[:,3].reshape(-1,1)


data_u10 = torch.cat([t10, x10, u10],1)
np.savetxt('timo_data_u10_inverse.csv' , [p for p in data_u10.detach().numpy()],  delimiter=',' , fmt='%s')
data_p10 = torch.cat([t10, x10, p10],1)
np.savetxt('timo_data_p10_inverse.csv' , [p for p in data_p10.detach().numpy()],  delimiter=',' , fmt='%s')
data_q10 = torch.cat([t10, x10, q10],1)
np.savetxt('timo_data_q10_inverse.csv' , [p for p in data_q10.detach().numpy()],  delimiter=',' , fmt='%s')
data_r10 = torch.cat([t10, x10, r10],1)
np.savetxt('timo_data_r10_inverse.csv' , [p for p in data_r10.detach().numpy()],  delimiter=',' , fmt='%s')

x11 = 2.4*torch.ones(100000).reshape(-1,1)
t11 =torch.rand(100000).reshape(-1,1)
test11 = torch.cat([x11, t11],1)

w11= my_network(test11)
u11= w11[:,0].reshape(-1,1)
p11= w11[:,1].reshape(-1,1)
q11 = w11[:,2].reshape(-1,1)
r11 = w11[:,3].reshape(-1,1)

data_u11 = torch.cat([t11, x11, u11],1)
np.savetxt('timo_data_u11_inverse.csv' , [p for p in data_u11.detach().numpy()],  delimiter=',' , fmt='%s')
data_p11 = torch.cat([t11, x11, p11],1)
np.savetxt('timo_data_p11_inverse.csv' , [p for p in data_p11.detach().numpy()],  delimiter=',' , fmt='%s')
data_q11 = torch.cat([t11, x11, q11],1)
np.savetxt('timo_data_q11_inverse.csv' , [p for p in data_q11.detach().numpy()],  delimiter=',' , fmt='%s')
data_r11 = torch.cat([t11, x11, r11],1)
np.savetxt('timo_data_r11_inverse.csv' , [p for p in data_r11.detach().numpy()],  delimiter=',' , fmt='%s')


x12 = 2.6*torch.ones(100000).reshape(-1,1)
t12 =torch.rand(100000).reshape(-1,1)
test12 = torch.cat([x12, t12],1)

w12= my_network(test12)
u12= w12[:,0].reshape(-1,1)
p12= w12[:,1].reshape(-1,1)
q12 = w12[:,2].reshape(-1,1)
r12 = w12[:,3].reshape(-1,1)

data_u12 = torch.cat([t12, x12, u12],1)
np.savetxt('timo_data_u12_inverse.csv' , [p for p in data_u12.detach().numpy()],  delimiter=',' , fmt='%s')
data_p12 = torch.cat([t11, x11, p12],1)
np.savetxt('timo_data_p12_inverse.csv' , [p for p in data_p12.detach().numpy()],  delimiter=',' , fmt='%s')
data_q12 = torch.cat([t12, x12, q12],1)
np.savetxt('timo_data_q12_inverse.csv' , [p for p in data_q12.detach().numpy()],  delimiter=',' , fmt='%s')
data_r12 = torch.cat([t12, x12, r12],1)
np.savetxt('timo_data_r12_inverse.csv' , [p for p in data_r12.detach().numpy()],  delimiter=',' , fmt='%s')

x13 = 2.8*torch.ones(100000).reshape(-1,1)
t13 =torch.rand(100000).reshape(-1,1)
test13 = torch.cat([x12, t12],1)

w13= my_network(test13)
u13= w13[:,0].reshape(-1,1)
p13= w13[:,1].reshape(-1,1)
q13 = w13[:,2].reshape(-1,1)
r13 = w13[:,3].reshape(-1,1)

data_u13 = torch.cat([t13, x13, u13],1)
np.savetxt('timo_data_u13_inverse.csv' , [p for p in data_u13.detach().numpy()],  delimiter=',' , fmt='%s')
data_p13 = torch.cat([t13, x13, p13],1)
np.savetxt('timo_data_p13_inverse.csv' , [p for p in data_p13.detach().numpy()],  delimiter=',' , fmt='%s')
data_q13 = torch.cat([t13, x13, q13],1)
np.savetxt('timo_data_q13_inverse.csv' , [p for p in data_q13.detach().numpy()],  delimiter=',' , fmt='%s')
data_r13 = torch.cat([t13, x13, r13],1)
np.savetxt('timo_data_r13_inverse.csv' , [p for p in data_r13.detach().numpy()],  delimiter=',' , fmt='%s')

x14 = 3*torch.ones(100000).reshape(-1,1)
t14 =torch.rand(100000).reshape(-1,1)
test14 = torch.cat([x14, t14],1)

w14= my_network(test14)
u14= w14[:,0].reshape(-1,1)
p14= w14[:,1].reshape(-1,1)
q14 = w14[:,2].reshape(-1,1)
r14 = w14[:,3].reshape(-1,1)

data_u14 = torch.cat([t14, x14, u14],1)
np.savetxt('timo_data_u14_inverse.csv' , [p for p in data_u14.detach().numpy()],  delimiter=',' , fmt='%s')
data_p14 = torch.cat([t14, x14, p14],1)
np.savetxt('timo_data_p14_inverse.csv' , [p for p in data_p14.detach().numpy()],  delimiter=',' , fmt='%s')
data_q14 = torch.cat([t13, x13, q13],1)
np.savetxt('timo_data_q14_inverse.csv' , [p for p in data_q14.detach().numpy()],  delimiter=',' , fmt='%s')
data_r14 = torch.cat([t14, x14, r14],1)
np.savetxt('timo_data_r14_inverse.csv' , [p for p in data_r14.detach().numpy()],  delimiter=',' , fmt='%s')
