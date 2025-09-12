import torch
import numpy as np

# ——— Core ———
SEED = 42
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Model
MODEL = 'TNO3d' # 'TNO3d   , FNO4d
# '
# ——— Geometry & Time (match MATLAB) ———
GRID_RESOLUTION = 32
L_DOMAIN = 2.0
DX = L_DOMAIN / GRID_RESOLUTION   # = 0.0625

DT = 1e-4
TOTAL_TIME_STEPS = 100            # Nt
TIME_END = DT * TOTAL_TIME_STEPS  # 0.01
SAVED_STEPS = TOTAL_TIME_STEPS + 1  # Nt+1 = 101

# ——— Physics (match MATLAB Allen–Cahn) ———
# u_t = Δu - (1/eps^2)*(u^3 - u)
EPSILON_PARAM = 0.1
EPS2 = EPSILON_PARAM**2

# ——— Data ———
MAT_DATA_PATH = "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/AC3D_32_600_grf3d.mat"
T_IN_CHANNELS = 4         # input window length for FNO4d
N_TRAIN = 100
N_TEST = 25

# ——— FNO4d ———
MODES = 12
WIDTH = 12 # 32
WIDTH_Q = 12 # 32
WIDTH_H = 12
N_LAYERS = 2 #4

# ——— Training ———
EPOCHS = 50 # 50 # 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PDE_WEIGHT = 0.25       # 25% physics
USE_AMP = False            # mixed precision for speed

# ——— Debug print scaling (to mimic your PINNs prints) ———
DEBUG_MU_SCALE = 1.0

# ——— Eval/plots ———
EVAL_TIME_FRAMES = [0, 50, 90]
