
import numpy as np

# General Setting
gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1300 #1600 # 4000
nTest = 200  # 400
batch_size = 20# 25 #100
learning_rate = 0.005
weight_decay = 1e-4
epochs = 30 # 1000
iterations = epochs * (nTrain // batch_size)
modes = 14 # 12
width = 12 #32
width_q = width #32
width_h = width // 2 # width # 32
n_layers = 2

# Discretization
s = 32
T_in = 1
T_out = 100 # 91

# Training Setting
normalized = True
training = True  # True  # True
load_model = False #True  # False  # False

# Database
parent_dir = './data/'
#matlab_dataset = 'MBE3D_2000_Nt_101_Nx_32.mat'
matlab_dataset = 'MBE3D_Augmented_2000_Nt_101_Nx_32.mat'


# Plotting

# In configs/config_SH3D_TNO3d.py (Add these or ensure they exist)
Lx = 2*np.pi # 1.0 # Domain size (assuming Lx=Ly=Lz based on MATLAB)
Ly =Lx
Lz= Lx


index = 62  # 24 # 62
domain = [-Lx/2, Lx/2]
#time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]

time_steps = [0, 50, 90]

### Hybrid method

# Time Discretization (from MATLAB)
dt_sim = 0.001 # Simulation time step
dt_simulation = 0.001
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
pde_loss_scaler = 1.5e0
###########################
# ... rest of config ...