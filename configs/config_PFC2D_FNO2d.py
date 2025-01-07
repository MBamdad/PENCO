import numpy as np

# General Setting
gpu_number = 'cuda:2'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 4000
nTest = 400
batch_size = 100
learning_rate = 0.005
weight_decay = 1e-4
epochs = 100
iterations = epochs * (nTrain // batch_size)
modes = 14
width = 16
width_q = 16
width_h = 16
n_layers = 4

# Discretization
s = 64
T_in = 1
T_out = 10

# Training Setting
normalized = True
training = True  # True
load_model = False  # False

# Database
parent_dir = './data/'
matlab_dataset = 'PFC2D_4440_Nt_101_Nx_64.mat'

# Plotting
index = 386  # 24  # 62  3  244
domain = [-np.pi, np.pi]
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
# time_steps = [0, 2, 4, 6, 8, 9]
