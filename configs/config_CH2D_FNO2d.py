import numpy as np

# General Setting
gpu_number = 'cuda'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 4000
nTest = 400
batch_size = 10
learning_rate = 0.001
weight_decay = 1e-4
epochs = 1000
iterations = epochs * (nTrain // batch_size)
modes = 16
width = 32
width_q = 2 * width
width_h = 0
n_layers = 4

# Discretization
s = 64
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = False  # True
load_model = False  # False

# Database
parent_dir = './data/'
matlab_dataset = 'CH2D_4400_Nt_101_Nx_64.mat'

# Plotting
index = 72
domain = [-np.pi, np.pi]
time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
# time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#               54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
