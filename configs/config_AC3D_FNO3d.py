import numpy as np

# General Setting
gpu_number = 'cuda'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1200 # 900 # 1000
nTest = 300 # 100 # 100
batch_size = 20 # 50 # 20 #5 # 25
learning_rate = 0.001
weight_decay = 1e-4
epochs = 25 # 50 # 100 # 900  # 100
iterations = epochs * (nTrain // batch_size)
modes =  14 # 8 # last time modes =  8
width =  12 # 32 # last time width =  32
width_q = width
width_h = width // 2 # width // 4 # last time
n_layers = 2

# Discretization
s = 32
T_in = 1
T_out = 100 # 100

# Training Setting
normalized = True
training = True  # False #  False
load_model = False # True  #  True  # True

# Database
parent_dir = './data/'
matlab_dataset = 'AC3D_32_1500_grf3d.mat'


# Plotting
# In configs/config_SH3D_TNO3d.py (Add these or ensure they exist)
Lx = 5 # np.pi # 1.0 # Domain size (assuming Lx=Ly=Lz based on MATLAB)
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
epsilon = 0.1
#pde_weight = 0.3 # Example: 30% physics loss
pde_weight = 0.5 # Example: 70% physics loss

# Learning Rate Scheduler Parameters (for StepLR)
scheduler_step = 20  # Decay learning rate every 20 epochs
scheduler_gamma = 0.5 # Multiply learning rate by 0.5 each time
pde_loss_scaler = 1e-3