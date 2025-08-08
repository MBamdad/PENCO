"""
Device:  cuda:0
model = TNO2d_PFC2D_S64_T1to100_width32_modes12_q32_h16.pt
number of epoch = 1000
batch size = 100
nTrain = 4000
nTest = 400
learning_rate = 0.005
n_layers = 4
width_q = 32
width_h = 16
2539703

The average testing error is 0.027059923857450485
Std. deviation of testing error is 0.01980009116232395
----------------------------------------------
model = TNO2d_PFC2D_S64_T1to50_width16_modes12_q32_h16.pt
number of epoch = 200
batch size = 100
nTrain = 4000
nTest = 400
learning_rate = 0.005
n_layers = 4
width_q = 32
width_h = 16
651059

The average testing error is 0.005057875066995621
Std. deviation of testing error is 0.0021678071934729815
----------------------------------------------


"""

import numpy as np

# General SettingnTrain = 1200 # 5250 # 7500

gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1200
nTest = 300
batch_size = 20 # 50 # 100
learning_rate = 0.001
weight_decay = 1e-4 # 1e-4
epochs = 30 # 50
iterations = epochs * (nTrain // batch_size)
modes = 14 #12
width = 12 # 16 # 32
width_q = width # 32
width_h = width//2 # 16
n_layers = 2

# Discretization
s = 32 # 64 # 64
T_in = 1
T_out = 91 # 100

# Training Setting
normalized = True
training = True # False  # True
load_model = False # True # False # True  # False

# Database
parent_dir = './data/'
matlab_dataset = 'PFC3D_1500_Nt_101_Nx_32.mat'


# Plotting
index = 200  # 110  # 200
domain = [-np.pi, np.pi]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
time_steps = [0, 50, 90]


### Hybrid method
# Time Discretization (from MATLAB)

dt_sim = 0.005 # Simulation time step
dt_simulation = 0.005
Nt = 100 # Total simulation steps
num_saved_steps = 101 # Number of saved steps (includes t=0)
ns = Nt / (num_saved_steps - 1) # Interval between saved steps
dt_model = ns * dt_sim # Effective time step between model outputs

# PDE Parameters
epsilon = 0.5
#pde_weight = 0.3 # Example: 30% physics loss
pde_weight = 0.5 # Example: 70% physics loss

# Learning Rate Scheduler Parameters (for StepLR)
scheduler_step = 20  # Decay learning rate every 20 epochs
scheduler_gamma = 0.5 # Multiply learning rate by 0.5 each time
pde_loss_scaler = 1e-3
