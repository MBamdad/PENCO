import numpy as np

# General Setting
gpu_number = 'cuda'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1000 # 900 # 1000
nTest = 300 # 100 # 100
batch_size = 50 # 20 #5 # 25
learning_rate = 0.001
weight_decay = 1e-4
epochs = 50 # 100 # 900  # 100
iterations = epochs * (nTrain // batch_size)
modes =  14 # 8 # last time modes =  8
width =  12 # 32 # last time width =  32
width_q = width
width_h = width // 2 # width // 4 # last time
n_layers = 4

# Discretization
s = 32
T_in = 1
T_out = 20 # 100

# Training Setting
normalized = True
training = True # False  # False
load_model = False # True  # True

# Database
parent_dir = './data/'
# matlab_dataset = 'AC3D_1200_Nt_101_Nx_32.mat'
#matlab_dataset = 'AC3D_32_1000.mat'
matlab_dataset = 'AC3D_32_1300.mat'
# Plotting
index = 9 # 12
domain = [-np.pi, np.pi]
# time_steps = [29, 69]
#time_steps = [39, 49, 59, 69, 79, 89, 99]
# time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#               54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
#time_steps = [39, 59, 79]
time_steps = [0, 9, 19]