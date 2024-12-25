
import numpy as np

# General Setting
gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 4000
nTest = 400
batch_size = 10
learning_rate = 0.001
weight_decay = 1e-4
epochs = 100  # 1000
iterations = epochs * (nTrain // batch_size)
modes = 18
width = 40  # 64
width_q = 40  # 40
width_h = 40  # 32
n_layers = 6  # 6

# Discretization
s = 64
T_in = 1
T_out = 80

# Training Setting
normalized = True
training = True  # True
load_model = False  # False

# Database
parent_dir = './data/'
matlab_dataset = 'CH2DNL_4400_Nt_101_Nx_64.mat'

# Plotting
index = 62  # 24 # 62
domain = [-np.pi, np.pi]
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
