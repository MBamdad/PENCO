import torch
import numpy as np
from pathlib import Path

# ——— Core ———
SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ——— Problem selector ———
# One of: 'AC3D', 'CH3D', 'SH3D', 'MBE3D', 'PFC3D'
PROBLEM = 'CH3D'

# ——— Model ———
MODEL = 'TNO3d'  # 'TNO3d' or 'FNO4d'

# Central registry of per-problem settings (geometry, time, epsilon, data path)
# NOTE: adjust MAT_DATA_PATHs to your actual files/locations.
PROBLEM_SPECS = {
    'AC3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT=1e-4,                  # your current default
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/AC3D_32_600_grf3d.mat",
        CH_LINEAR_COEF=3.0,   # <-- matches your dataset,
    ),
    'CH3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT=5e-4,                  # from your spec Δt ≈ 0.0005
        TOTAL_TIME_STEPS=100,     # feel free to change
        EPSILON_PARAM=0.05,
        MAT_DATA_PATH = "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/CH3D_1500_Nt_101_Nx_32.mat",
    ),
    'SH3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=15.0,
        DT=5e-2,                  # 0.05
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.15,
        MAT_DATA_PATH="/scratch/.../SH3D_32_xxx.mat",
    ),
    'MBE3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2*np.pi,
        DT=1e-3,                  # 0.001
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH="/scratch/.../MBE3D_32_xxx.mat",
    ),
    'PFC3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=10*np.pi,
        DT=5e-3,                  # 0.005
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.5,
        MAT_DATA_PATH="/scratch/.../PFC3D_32_xxx.mat",
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



# ——— FNO4d / TNO3d ———
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
EVAL_TIME_FRAMES = [0, 50, 90]
