import torch
import numpy as np
from pathlib import Path
import random
import torch as _torch

# ——— Core ———
SEED = 42

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# ——— Problem selector ———
# One of: 'AC3D', 'CH3D', 'SH3D', 'MBE3D', 'PFC3D'
PROBLEM = 'PFC3D'   # <- set here when you want Swift–Hohenberg
# ——— Model ———
MODEL = 'FNO4d'  # 'TNO3d' or 'FNO4d'

PROBLEM_SPECS = {
    'AC3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT=1e-4,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH="/scratch/noqu8762/PENCOO/data/AC3D_32_250_grf3d.mat",
        CH_LINEAR_COEF=3.0,
    ),
    'CH3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT= 1e-3, #5e-4, #5e-4, # 5e-3,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.05,
        #MAT_DATA_PATH = "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/CH3D_500_Nt_101_Nx_32.mat", # correct, dt= 0.005
        MAT_DATA_PATH= '/scratch/noqu8762/PENCOO/data/CH3D_250_Nt_101_Nx_32.mat', # dt = 0.001
    ),
    'SH3D': dict(
        GRID_RESOLUTION=32,   # Nx = Ny = Nz
        L_DOMAIN=15.0,
        DT = 5e-2,              # 0.05
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.15,   # appears as (1 - eps) in the SH linear term
        MAT_DATA_PATH="/scratch/noqu8762/PENCOO/data/SH3D_grf3d_ff_250_Nt_101_Nx_32.mat",
        #MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/SH3D_grf3d_ff_250_Nt_101_Nx_32.mat",
    ),

    'MBE3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2*np.pi,
        DT=5e-3,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH = '/scratch/noqu8762/PENCOO/data/MBE3D_Augmented_250_Nt_101_Nx_32.mat', # dt=0.005
        #MAT_DATA_PATH = '/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/MBE3D_Augmented_250_Nt_101_Nx_32.mat', # dt=0.005
    ),
    'PFC3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=10*np.pi,
        DT=1e-2, # 1e-2, old and valid
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.5,
        MAT_DATA_PATH="/scratch/noqu8762/PENCOO/data/PFC3D_Augmented_250_Nt_101_Nx_32.mat", # dt=0.01
        #MAT_DATA_PATH ="/data/PFC3D_Augmented_250_Nt_101_Nx_32_dt05.mat", # dt=0.05
        #MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/PFC3D_Augmented_250_Nt_101_Nx_32.mat", # dt=0.01
    ),
}

def seed_everything(seed: int = SEED):
    print(f"[Seed] Using seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN for reproducibility
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False

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
PHYS_MAX_SCALE = 2.0
# ——— Data ———
MAT_DATA_PATH = _spec['MAT_DATA_PATH']


SCALE_STEPS_WITH_NTRAIN = True   # set True to mimic "beam behavior"
use_lbfgs = False # True
N_TRAIN_REF = 50                 # reference N_TRAIN that matches your current STEPS_PER_EPOCH

if PROBLEM == 'MBE3D':
    STEPS_PER_EPOCH = 30  # MBE  # pick once; same training budget regardless of N_TRAIN
elif PROBLEM == 'PFC3D':
    STEPS_PER_EPOCH = 10  # PFC
elif PROBLEM == 'SH3D':
    STEPS_PER_EPOCH = 5  # SH3D
elif PROBLEM == 'CH3D':
    STEPS_PER_EPOCH = 30 # 80  # CH3D
elif PROBLEM == 'AC3D':
    STEPS_PER_EPOCH = 10  # AC3D

else:
    print('Enter the right PROBLEM !')


N_TEST_FIXED = 50 #50 AC, 100 # 100 # 100             # <- constant, test set size is fixed now
TEST_MODE = 'manual'   # or 'manual'
TEST_PICK = 1       # 0 spherical   # only used if TEST_MODE == 'manual'

PURE_PHYSICS_USE_ALL = True    # <- when PDE_WEIGHT==1.0, ignore N_TRAIN for train


MODES = 10 # 12  # Or whatever value was used during training
WIDTH = 10 # 12 # This is the most likely one to change

if PROBLEM == 'MBE3D': # old
    WIDTH_Q = 11
    WIDTH_H = 11
else:
    WIDTH_Q = 10
    WIDTH_H = 10

N_LAYERS = 2

# ——— Training ———
if PROBLEM == 'SH3D': # old
    EPOCHS = 150
else:
    EPOCHS = 50

BATCH_SIZE = 8
if PROBLEM == 'MBE3D':
    LEARNING_RATE = 5e-4
else:
    LEARNING_RATE = 1e-3

T_IN_CHANNELS = 4 # number of past frames the model sees
T_OUT = 1 # determines how much future the model predicts


WEIGHT_DECAY = 1e-5
PDE_WEIGHT = 0.25

N_TRAIN = 200
N_TEST = max(1, N_TRAIN // 4)

# ——— Debug print scaling ———
DEBUG_MU_SCALE = 1.0

# ——— Eval/plots ———
EVAL_TIME_FRAMES = [0, 50, 100]