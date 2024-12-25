"""
TNO2d_CH2DNL_S64_T1to100_width32_modes16, epoch = 100, batch = 20, width_q = 32, width_h = 32
n_layers = 4
The average testing error is 0.10818369686603546
Std. deviation of testing error is 0.023711388930678368
n_layers = 6
The average testing error is 0.08396226912736893
Std. deviation of testing error is 0.020571144297719002
n_layers = 8
The average testing error is 0.09714231640100479
Std. deviation of testing error is 0.0222222488373518
n_layers = 10
The average testing error is 0.9998834729194641
Std. deviation of testing error is 0.0012830810155719519
__________________________________________________________________
TNO2d_CH2DNL_S64_T1to100_width32_modes16, epoch = 100, batch = 20, width_q = 32, width_h = 32, n_layers = 6
width_h = 32
The average testing error is 0.08396226912736893
Std. deviation of testing error is 0.020571144297719002
width_h = 16
The average testing error is 0.08495531231164932
Std. deviation of testing error is 0.01994013786315918
width_h = 20
The average testing error is 0.08499683439731598
Std. deviation of testing error is 0.020479857921600342
width_h = 40
The average testing error is 0.08685333281755447
Std. deviation of testing error is 0.020362991839647293
width_h = 64
The average testing error is 0.08670475333929062
Std. deviation of testing error is 0.021706586703658104
__________________________________________________________________
TNO2d_CH2DNL_S64_T1to100_width32_modes16, epoch = 100, batch = 20, width_q = 32, width_h = 32, n_layers = 6
width = 32
The average testing error is 0.08396226912736893
Std. deviation of testing error is 0.020571144297719002

width = 16
The average testing error is 0.12493790686130524
Std. deviation of testing error is 0.02295171096920967
width = 20
The average testing error is 0.10955392569303513
Std. deviation of testing error is 0.022093135863542557
width = 40
The average testing error is 0.0751548632979393
Std. deviation of testing error is 0.02070842683315277
width = 64
The average testing error is 0.06826775521039963
Std. deviation of testing error is 0.019642595201730728
__________________________________________________________________
TNO2d_CH2DNL_S64_T1to100_width64_modes16, epoch = 100, batch = 20, width = 64, width_h = 32, n_layers = 6
width_q = 32
The average testing error is 0.06826775521039963
Std. deviation of testing error is 0.019642595201730728

width_q = 20
The average testing error is 0.07036576420068741
Std. deviation of testing error is 0.01962932199239731
width_q = 32
The average testing error is 0.06826775521039963
Std. deviation of testing error is 0.019642595201730728
width_q = 40
The average testing error is 0.06606774032115936
Std. deviation of testing error is 0.018914170563220978
width_q = 64
The average testing error is 0.06615814566612244
Std. deviation of testing error is 0.018680864945054054
__________________________________________________________________
TNO2d_CH2DNL_S64_T1to100_width64_modes16, epoch = 100, width = 64, width_h = 32, width_q = 40, n_layers = 6
batch = 20
The average testing error is 0.06606774032115936
Std. deviation of testing error is 0.018914170563220978
batch = 10
The average testing error is 0.06392485648393631
Std. deviation of testing error is 0.017472662031650543
batch = 25
The average testing error is 0.07051659375429153
Std. deviation of testing error is 0.019806643947958946
batch = 50
The average testing error is 0.08658520877361298
Std. deviation of testing error is 0.022557144984602928
batch = 100
The average testing error is 0.10399674624204636
Std. deviation of testing error is 0.023524906486272812
__________________________________________________________________
TNO2d_CH2DNL_S64_T1to100_width64_modes16, epoch = 100, width = 64, width_h = 32, width_q = 40, n_layers = 6, batch = 20
modes = 16
The average testing error is 0.06606774032115936
Std. deviation of testing error is 0.018914170563220978
modes = 8

modes = 10
The average testing error is 0.1149306520819664
Std. deviation of testing error is 0.026237860321998596
modes = 12
The average testing error is 0.07895160466432571
Std. deviation of testing error is 0.021178798750042915
modes = 20
The average testing error is 0.06290720403194427
Std. deviation of testing error is 0.019949892535805702
__________________________________________________________________
__________________________________________________________________
T = 50,
model = TNO2d_CH2DNL_S64_T1to50_width32_modes16.pt
number of epoch = 200
batch size = 20
nTrain = 4000
nTest = 400
learning_rate = 0.001
n_layers = 4
width_q = 32
width_h = 32
The average testing error is 0.029468493536114693
Std. deviation of testing error is 0.009010270237922668
_______________________________________________________
Check for better solution when T=100
model = TNO2d_CH2DNL_S64_T1to100_width32_modes16.pt
number of epoch = 1000
batch size = 20
nTrain = 4000
nTest = 400
learning_rate = 0.001
n_layers = 6
width_q = 40
width_h = 32
6456295
The average testing error is 0.055347371846437454
Std. deviation of testing error is 0.0164803434163332
_______________________________________________________________
Final:
model = TNO2d_CH2DNL_S64_T1to100_width40_modes18.pt
number of epoch = 1000
batch size = 20
nTrain = 4000
nTest = 400
learning_rate = 0.001
n_layers = 6
width_q = 40
width_h = 40
Found saved dataset at ./data/CH2DNL_4400_Nt_101_Nx_64.pt
torch.Size([4400, 64, 64, 1])
torch.Size([4400, 64, 64, 100])
12651359
The average testing error is 0.04985285922884941
Std. deviation of testing error is 0.014996719546616077
Min testing error is 0.024426046758890152 at index 197
Max testing error is 0.14625325798988342 at index 236
Mode of testing errors is 0.024426046758890152 appearing 197 times at indices 197

"""
import numpy as np

# General Setting
gpu_number = 'cuda:0'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 4000
nTest = 400
batch_size = 20
learning_rate = 0.001
weight_decay = 1e-4
epochs = 1000  # 1000
iterations = epochs * (nTrain // batch_size)
modes = 18
width = 40  # 64
width_q = 40  # 40
width_h = 32  # 32
n_layers = 6  # 6

# Discretization
s = 64
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = False  # True
load_model = True  # False

# Database
parent_dir = './data/'
matlab_dataset = 'CH2DNL_4400_Nt_101_Nx_64.mat'

# Plotting
index = 197  # 24  # 236  # 197  # 24
domain = [-np.pi, np.pi]
# time_steps = [29, 35, 39, 45, 49]
# time_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
              54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
