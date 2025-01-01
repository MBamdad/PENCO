import numpy as np

# General Setting
gpu_number = 'cuda:3'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 10000
nTest = 1000
batch_size = 20
learning_rate = 0.005
weight_decay = 1e-4
epochs = 100
iterations = epochs * (nTrain // batch_size)
modes = 12
width = 64
width_q = 32
width_h = 16
n_layers = 7

# Discretization
s = 64
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = False#True  # True
load_model = True#False  # False

# Database
parent_dir = './data/'
matlab_dataset = 'SH2D_11100_Nt_101_Nx_64.mat'

# Plotting
index = 24  # 62
domain = [-np.pi, np.pi]
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
# time_steps = [0, 2, 4, 6, 8, 9]
