import numpy as np

# General Setting
gpu_number = '1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1000
nTest = 100
batch_size = 10
learning_rate = 0.001
weight_decay = 1e-4
epochs = 1#500
iterations = epochs * (nTrain // batch_size)
modes = 12
width = 32

# Discretization
s = 64
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = True
load_model = False

# Database
parent_dir = './data/'
matlab_dataset = 'AC2D_2000_Nt_101_Nx_64.mat'


# Plotting
index = 50
domain = [-np.pi, np.pi]
time_steps = [5, 10, -1]
