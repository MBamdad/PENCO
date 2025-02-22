import numpy as np

# General Setting
gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 600 #900 # 500 # 6000 # 8000# 1200 #1200 # 7500
nTest =  200 # 300 # 100 # 2000 # 4000 #300 # 300 #500
batch_size = 3 # 20 # 50 # 5 # 50
learning_rate = 0.001 # 0.0001 # 0.001 # 0.005 # 0.001
weight_decay = 1e-4 # 1e-3 # 1e-4
epochs = 500 # 1000 # 25 # 100 # 1000
iterations = epochs * (nTrain // batch_size)
modes = 10 #12 # 14 # 16 # 10 # 16
width = 8 #12 #14 # 12 # 16 # 32
width_q = width # 2 * width #
width_h = width//2 # width//4 # width #
n_layers = 4 # 4 # 5 # 5 # 8

# Discretization

s = 64 #32 # 64
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = True # False # True # False  # True
load_model = False # True # False # False #True

# Database
parent_dir = './data/'
#matlab_dataset = 'CH3D_8000_Nt_101_Nx_32.mat'
#matlab_dataset = 'CH3D_1500_Nt_101_Nx_32_old.mat'
#matlab_dataset = 'CH3D_1200_Nt_101_Nx_64.mat' # (average testing error: 0.059 for n_train = 500)
matlab_dataset = 'CH3D_800_Nt_101_Nx_64.mat'
#matlab_dataset = 'CH3D_12000_Nt_101_Nx_32.mat'
#matlab_dataset = 'CH3D_5000_Nt_101_Nx_32.mat'
#matlab_dataset = 'CH3D_8000_Nt_101_Nx_32.mat'
#matlab_dataset = 'CH3D_2000_Nt_101_Nx_64.mat'
#matlab_dataset = 'CH3D_1300_Nt_101_Nx_32.mat'

# Plotting
index = 62  # 24 # 62
domain = [-np.pi, np.pi]
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
#time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
#time_steps = [39, 49, 59, 69, 79, 89, 99]
time_steps = [39, 59, 79]