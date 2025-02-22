
import numpy as np

# General Setting
gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1200 #1600 # 4000
nTest = 300  # 400
batch_size = 25 #100
learning_rate = 0.005
weight_decay = 1e-4
epochs = 50 # 1000
iterations = epochs * (nTrain // batch_size)
modes = 14 # 12
width = 12 #32
width_q = width #32
width_h = width // 2 # width # 32
n_layers = 4

# Discretization
s = 32
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = True  # True  # True
load_model = False #True  # False  # False

# Database
parent_dir = './data/'
#matlab_dataset = 'MBE3D_2000_Nt_101_Nx_32.mat'
matlab_dataset = 'MBE3D_1500_Nt_101_Nx_32.mat'
# Plotting
index = 9  # 72
domain = [-np.pi, np.pi]
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
#time_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
time_steps = [39, 59, 79,]