import torch
import numpy as np
from pathlib import Path

# ——— Core ———
SEED = 42
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# ——— Problem selector ———
# One of: 'AC3D', 'CH3D', 'SH3D', 'MBE3D', 'PFC3D'
PROBLEM = 'CH3D'   # <- set here when you want Swift–Hohenberg

# ——— Model ———
MODEL = 'FNO4d'  # 'TNO3d' or 'FNO4d'

PROBLEM_SPECS = {
    'AC3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT=1e-4,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/AC3D_32_600_grf3d.mat",
        CH_LINEAR_COEF=3.0,
    ),
    'CH3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT=5e-3,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.05,
        #MAT_DATA_PATH = "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/CH3D_1500_Nt_101_Nx_32.mat",
        MAT_DATA_PATH = "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/CH3D_500_Nt_101_Nx_32.mat",
    ),
    'SH3D': dict(
        GRID_RESOLUTION=32,   # Nx = Ny = Nz
        L_DOMAIN=15.0,
        DT = 5e-2,              # 0.05
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.15,   # appears as (1 - eps) in the SH linear term
        MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/SH3D_grf3d_ff_500_Nt_101_Nx_32.mat",
    ),

    'MBE3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2*np.pi,
        DT=5e-3,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/MBE3D_Augmented_600_Nt_101_Nx_32.mat",
    ),
    'PFC3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=10*np.pi,
        DT=1e-2,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.5,
        MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/PFC3D_Augmented_600_Nt_101_Nx_32.mat",
    ),
}

# Load active problem spec
_spec = PROBLEM_SPECS[PROBLEM]

# For CH3D only: u_t = ∇²(u^3 - alpha * u) - ε^2 ∇^4 u
CH_LINEAR_COEF = PROBLEM_SPECS.get('CH3D', {}).get('CH_LINEAR_COEF', 3.0)

# ——— Geometry & Time (per PROBLEM) ———
GRID_RESOLUTION = _spec['GRID_RESOLUTION']
L_DOMAIN = _spec['L_DOMAIN']
DX = L_DOMAIN / GRID_RESOLUTION

DT = _spec['DT']
TOTAL_TIME_STEPS = _spec['TOTAL_TIME_STEPS']
TIME_END = DT * TOTAL_TIME_STEPS
SAVED_STEPS = TOTAL_TIME_STEPS + 1

# ——— Physics ———
EPSILON_PARAM = _spec['EPSILON_PARAM']
EPS2 = EPSILON_PARAM ** 2

# ——— Data ———
MAT_DATA_PATH = _spec['MAT_DATA_PATH']
T_IN_CHANNELS = 4
N_TRAIN = 200
N_TEST = max(1, N_TRAIN // 4)
SCALE_STEPS_WITH_NTRAIN = True   # set True to mimic "beam behavior"
N_TRAIN_REF = 50                 # reference N_TRAIN that matches your current STEPS_PER_EPOCH

if PROBLEM == 'MBE3D':
    STEPS_PER_EPOCH = 25  # MBE  # pick once; same training budget regardless of N_TRAIN
elif PROBLEM == 'PFC3D':
    STEPS_PER_EPOCH = 5  # PFC
elif PROBLEM == 'SH3D':
    STEPS_PER_EPOCH = 10  # SH3D
elif PROBLEM == 'CH3D':
    STEPS_PER_EPOCH = 80  # CH3D
elif PROBLEM == 'AC3D':
    STEPS_PER_EPOCH = 10  # AC3D

else:
    print('Enter the right PROBLEM !')


N_TEST_FIXED = 50 # 100 # 100 # 100             # <- constant, test set size is fixed now
PURE_PHYSICS_USE_ALL = True    # <- when PDE_WEIGHT==1.0, ignore N_TRAIN for train

# ——— FNO4d / TNO3d ———
if PROBLEM == 'AC3D':
    MODES = 10 # 12  # Or whatever value was used during training
    WIDTH = 10 # 12 # This is the most likely one to change
else:
    MODES = 10
    WIDTH = 10

WIDTH_Q = 12
WIDTH_H = 12
N_LAYERS = 2

# ——— Training ———
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3 # 3e-4 --> CH3d 1e-3 --> AC3d
WEIGHT_DECAY = 1e-5
PDE_WEIGHT = 0.25

# ——— Debug print scaling ———
DEBUG_MU_SCALE = 1.0

# ——— Eval/plots ———
EVAL_TIME_FRAMES = [0, 50, 100]