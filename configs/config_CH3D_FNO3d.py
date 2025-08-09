import numpy as np

# General Setting
gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1300
nTest =  200
batch_size = 20 # 50
learning_rate = 0.001 # 0.0001 #
weight_decay = 1e-4 # 1e-3 # 1e-4
epochs = 30 # 100 # 500
iterations = epochs * (nTrain // batch_size)
modes = 14 # 10 #12 # 14 # 16 # 10 # 16
width = 12 # 8 #12 #14 # 12 # 16 # 32
width_q = width # 2 * width #
width_h = width//2 # width//4 # width #
n_layers = 2 # 4 # 5 # 5 # 8

# Discretization

s = 32 # 64 #32 # 64
T_in = 1
T_out = 100 # 91

# Training Setting
normalized = True
training = True # False # True # False  # True
load_model = False # True # False # False #True

# Database
parent_dir = './data/'
matlab_dataset = 'CH3D_1500_Nt_101_Nx_32.mat'



# Plotting
# In configs/config_SH3D_TNO3d.py (Add these or ensure they exist)
Lx = 2 # np.pi # 1.0 # Domain size (assuming Lx=Ly=Lz based on MATLAB)
Ly =Lx
Lz= Lx
index = 62  # 24 # 62
#domain = [-np.pi, np.pi] ######
domain = [-Lx/2, Lx/2]
#time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
time_steps = [0, 50, 90]


### Hybrid method

# Time Discretization (from MATLAB)
dt_sim = 0.0005 # Simulation time step
dt_simulation = 0.0005
Nt = 100 # Total simulation steps
num_saved_steps = 101 # Number of saved steps (includes t=0)
ns = Nt / (num_saved_steps - 1) # Interval between saved steps
dt_model = ns * dt_sim # Effective time step between model outputs

# PDE Parameters
epsilon = 0.05
#pde_weight = 0.3 # Example: 30% physics loss
pde_weight = 0.5 # Example: 70% physics loss

# Learning Rate Scheduler Parameters (for StepLR)
scheduler_step = 20  # Decay learning rate every 20 epochs
scheduler_gamma = 0.5 # Multiply learning rate by 0.5 each time
pde_loss_scaler = 1e-2

