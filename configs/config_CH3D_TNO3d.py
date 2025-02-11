import numpy as np

# General Setting
gpu_number = 'cuda'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1200 # 7500
nTest =  300 #500
batch_size = 50 # 5 # 50
learning_rate = 0.001 # 0.005 # 0.001
weight_decay = 1e-3 # 1e-3 # 1e-4
epochs = 25 # 100 # 1000
iterations = epochs * (nTrain // batch_size)
modes = 16 # 10 # 16
width = 16 # 32
width_q = width # 2 * width #
width_h = width//2 # width//4 # width #
n_layers = 5 # 5 # 8

# Discretization

s = 32 # 64
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = True # False  # True
load_model = True # False #True

# Database
parent_dir = './data/'
#matlab_dataset = 'CH3D_8000_Nt_101_Nx_32.mat'
matlab_dataset = 'CH3D_1500_Nt_101_Nx_32.mat'
#matlab_dataset = 'CH3D_2000_Nt_101_Nx_64.mat'
#matlab_dataset = 'CH3D_8000_Nt_101_Nx_32_compressed.npz'
# Plotting
index = 62  # 24 # 62
domain = [-np.pi, np.pi]
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
#time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
#time_steps = [39, 49, 59, 69, 79, 89, 99]
time_steps = [39, 59, 79]