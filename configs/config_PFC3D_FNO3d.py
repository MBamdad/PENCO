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
batch_size = 50 # 100
learning_rate = 0.005
weight_decay = 1e-4 # 1e-4
epochs = 50
iterations = epochs * (nTrain // batch_size)
modes = 14 #12
width = 12 # 16 # 32
width_q = width # 32
width_h = width//2 # 16
n_layers = 4

'''
tau = 315;
alpha = 115; 
'''

# Discretization
s = 32 # 64 # 64
T_in = 1
T_out = 20 # 100

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
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
#time_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
# time_steps = [0, 2, 4, 6, 8, 9]
#time_steps = [39, 59, 79]
time_steps = [0, 9, 19]