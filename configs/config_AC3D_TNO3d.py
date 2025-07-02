import numpy as np

# General Setting
gpu_number = 'cuda'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1000 # 900 # 1000
nTest = 300 # 100 # 100
batch_size = 10 # 50 # 20 #5 # 25
learning_rate = 0.001
weight_decay = 1e-4
epochs = 50 # 100 # 900  # 100
iterations = epochs * (nTrain // batch_size)
modes =  14 # 8 # last time modes =  8
width =  12 # 32 # last time width =  32
width_q = width
width_h = width // 2 # width // 4 # last time
n_layers = 4

# Discretization
s = 32
T_in = 1
T_out = 20 # 100

# Training Setting
normalized = True
training = True  # False #  False
load_model = False # True  #  True  # True

# Database
parent_dir = './data/'
# matlab_dataset = 'AC3D_1200_Nt_101_Nx_32.mat'
#matlab_dataset = 'AC3D_32_1000.mat'
matlab_dataset = 'AC3D_32_1300.mat'
# Plotting
index = 9 # 12
domain = [-np.pi, np.pi]
# time_steps = [29, 69]
#time_steps = [39, 49, 59, 69, 79, 89, 99]
# time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#               54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
#time_steps = [39, 59, 79]
time_steps = [0, 9, 19]


#############
Lx = np.pi            # Domain size from MATLAB
# Time Discretization Parameters (from AC3D MATLAB)
dt_sim = 0.0001     # Time step in the MATLAB simulation
Nt_sim = 50        # Total number of simulation steps in MATLAB
num_saved_steps_sim = 101 # Number of steps saved in the .mat file (Nt_sim + 1)
# ns_sim = Nt_sim / (num_saved_steps_sim - 1) if num_saved_steps_sim > 1 else 1.0 # Saving interval in sim steps
# dt_model = ns_sim * dt_sim # Effective time step between frames in your data/model output
# For AC3D, MATLAB code saves all Nt+1 steps. So ns_sim = 1.
dt_model = dt_sim # If T_out steps directly correspond to dt_sim steps after T_in
# PDE Parameters for Allen-Cahn (AC3D)
# From your MATLAB: epsilon = 0.1, Cahn = epsilon^2.
# The PDE implemented in calculate_pde_residual was:
# du/dt = Cahn_ac * laplacian(u) - (u^3 - u)
epsilon = 0.1  # The epsilon from your AC3D MATLAB script, PDE parameter
# PINN Specific Settings (if PINN_MODE is True in main.py)
pde_weight = 0.5   # Example: 50% physics loss, 70% data loss. Adjust as needed.
# PDE_LOSS_SCALER will be defined in main.py, but you might note its value here for reference

# Learning Rate Scheduler Parameters (for StepLR)
scheduler_step = 20  # Decay learning rate every 20 epochs
scheduler_gamma = 0.5 # Multiply learning rate by 0.5 each time

pde_loss_scaler = 1e-3