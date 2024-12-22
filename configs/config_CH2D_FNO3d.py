# T_out = 100 , width = 16, modes = 8, batch_size = 25, epochs = 1000 , param num = 8395505, nTrain = 4000, nTest = 400
# The average testing error is 0.0866110771894455
# Std. deviation of testing error is 0.07253412902355194
# Min testing error is 0.018533991649746895 at index 114
# Max testing error is 0.3846074938774109 at index 70
# Mode of testing errors is 0.018533991649746895 appearing 114 times at indices 114

import numpy as np

# General Setting
gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 4000
nTest = 400
batch_size = 25
learning_rate = 0.001
weight_decay = 1e-4
epochs = 1000
iterations = epochs * (nTrain // batch_size)
modes = 8
width = 16
n_layers = 8

# Discretization
s = 64
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = True  # True
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
