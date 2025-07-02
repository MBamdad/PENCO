import numpy as np

# General Setting
gpu_number = 'cuda:2'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1200 # 5250 # 7500
nTest = 300 # 2250 #500
batch_size = 20 # 50
learning_rate = 0.001 # 0.005 # 0.001
weight_decay = 1e-4 # 1e-4
epochs = 30 # 50 # 50 # 50 # 50 # 1000
iterations = epochs * (nTrain // batch_size)
modes =  14 # 16 # 16
width =  12 # 12 #32
width_q =   width # width # 2 * width #
width_h = width//2  # width//4 # width #
n_layers = 2 # 8

# Discretization
#s = 32 # 32 # 64
#T_in = 1
#T_out = 20 # 100
# Discretization
s = 32 # 80         # CRITICAL: Must match Nx, Ny, Nz from MATLAB (which is 80)
T_in = 1       # CRITICAL: Use the first time step (t=0) as input
T_out = 100 # 91 # 100 # 20     # CRITICAL: Predict the next 10 time steps. Total steps used = 1+10=11, which matches the data.

# Training Setting
normalized = True # False
training = True # False  #   True
load_model = False # True # False # False #True

# Database
parent_dir = './data/'

#matlab_dataset = 'SH3D_1500_Nt_101_Nx_3232.mat' # correct
#matlab_dataset = 'SH3D_1500_Nt_101_Nx_32.mat'
matlab_dataset = 'SH3D_grf3d_1500_Nt_101_Nx_32.mat' # Correct code # TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results
# Plotting

# In configs/config_SH3D_TNO3d.py (Add these or ensure they exist)
Lx = 10 # np.pi # 1.0 # Domain size (assuming Lx=Ly=Lz based on MATLAB)
Ly =Lx
Lz= Lx


index = 62  # 24 # 62
#domain = [-np.pi, np.pi] ######
domain = [-Lx/2, Lx/2]

# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
#time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
#time_steps = [39, 49, 59, 69, 79, 89, 99]
#time_steps = [39, 59, 79]
#time_steps = [0, 9, 19]
time_steps = [0, 50, 90]
####
### Hybrid method


#domain = [-Lx/2, Lx/2] # Assuming centered domain

# Time Discretization (from MATLAB)
dt_sim = 0.05 # Simulation time step
dt_simulation = 0.05
Nt = 100 # Total simulation steps
num_saved_steps = 101 # Number of saved steps (includes t=0)
ns = Nt / (num_saved_steps - 1) # Interval between saved steps
dt_model = ns * dt_sim # Effective time step between model outputs

# PDE Parameters
epsilon = 0.15
#pde_weight = 0.3 # Example: 30% physics loss
pde_weight = 0.4 # Example: 70% physics loss

# Learning Rate Scheduler Parameters (for StepLR)
scheduler_step = 20  # Decay learning rate every 20 epochs
scheduler_gamma = 0.5 # Multiply learning rate by 0.5 each time
pde_loss_scaler = 1e1
###########################
# ... rest of config ...