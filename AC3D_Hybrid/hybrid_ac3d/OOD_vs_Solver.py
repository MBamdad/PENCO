import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.io import savemat
from scipy.ndimage import zoom as nd_zoom
import time

# --- repo imports (your files) ---
import config as CFG
from networks import FNO4d, TNO3d
from functions import semi_implicit_step, semi_implicit_step_pfc, semi_implicit_step_mbe, semi_implicit_step_sh

# ======================================================================
# 0) GLOBAL SETTINGS
# ======================================================================
N_REPS_TIME = 5  # repetitions for timing (solver + models)

# -------------------------------------------------------------------
# 1) PROBLEM-SPECIFIC CONFIGURATIONS
# -------------------------------------------------------------------

# --- Checkpoints for AC3D ---
CKPTS_AC3D = {
    # method_label: (model_type, path)
    "FNO4d": ("FNO4d",
              "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/FNO4d_FNO4d_N200_pw0.00_E50.pt"),
    "MHNO": ("TNO3d",
             "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/TNO3d_MHNO_N200_pw0.00_E50.pt"),
    "PENCO-MHNO": ("TNO3d",
                   "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/TNO3d_PENCO_N200_pw0.50_E50.pt"),
    "PENCO-FNO": ("FNO4d",
                  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/FNO4d_PENCO_N200_pw0.50_E50.pt"),
    "PurePhysics": ("TNO3d",
                    "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/TNO3d_PurePhysics_N200_pw1.00_E50.pt"),
}

# --- Checkpoints for CH3D ---
CKPTS_CH3D = {
    # method_label: (model_type, path)
    "FNO4d": ("FNO4d",
              "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/CH3d_models/FNO4d_FNO4d_N200_pw0.00_E50.pt"),
    "MHNO": ("TNO3d",
             "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/CH3d_models/TNO3d_MHNO_N200_pw0.00_E50.pt"),
    "PENCO-MHNO": ("TNO3d",
                   "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/CH3d_models/TNO3d_PENCO_N200_pw0.25_E50.pt"),
    "PENCO-FNO": ("FNO4d",
                  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/CH3d_models/FNO4d_PENCO_N200_pw0.25_E50.pt"),
    "PurePhysics": ("TNO3d",
                    "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/CH3d_models/TNO3d_PurePhysics_N200_pw1.00_E50.pt"),
}

# --- Checkpoints for SH3D ---
CKPTS_SH3D = {
    # method_label: (model_type, path)
    "FNO4d": ("FNO4d",
              "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/FNO4d_FNO4d_N200_pw0.00_E50.pt"),
    "MHNO": ("TNO3d",
             "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/TNO3d_MHNO_N200_pw0.00_E50.pt"),
    "PENCO-MHNO": ("TNO3d",
                   "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/TNO3d_PENCO_N200_pw0.75_E50.pt"),
    "PENCO-FNO": ("FNO4d",
                  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/FNO4d_PENCO_N200_pw0.75_E50.pt"),
    "PurePhysics": ("TNO3d",
                    "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/TNO3d_PurePhysics_N200_pw1.00_E50.pt"),
}

# --- Checkpoints for PFC3D ---
CKPTS_PFC3D = {
    # method_label: (model_type, path)
    "FNO4d": ("FNO4d",
              "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/FNO4d_FNO4d_N200_pw0.00_E50.pt"),
    "MHNO": ("TNO3d",
             "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/TNO3d_MHNO_N200_pw0.00_E50.pt"),
    "PENCO-MHNO": ("TNO3d",
                   "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/TNO3d_PENCO_N200_pw0.75_E50.pt"),
    "PENCO-FNO": ("FNO4d",
                  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/FNO4d_PENCO_N200_pw0.75_E50.pt"),
    "PurePhysics": ("TNO3d",
                    "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/TNO3d_PurePhysics_N200_pw1.00_E50.pt"),
}

# --- Checkpoints for MBE3D ---
CKPTS_MBE3D = {
    # method_label: (model_type, path)
    "FNO4d": ("FNO4d",
              "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/MBE3d_models/FNO4d_FNO4d_N200_pw0.00_E50.pt"),
    "MHNO": ("TNO3d",
             "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/MBE3d_models/TNO3d_MHNO_N200_pw0.00_E50.pt"),
    "PENCO-MHNO": ("TNO3d",
                   "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/MBE3d_models/TNO3d_PENCO_N200_pw0.25_E50.pt"),
    "PENCO-FNO": ("FNO4d",
                  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/MBE3d_models/FNO4d_PENCO_N200_pw0.25_E50.pt"),
    "PurePhysics": ("TNO3d",
                    "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/MBE3d_models/TNO3d_PurePhysics_N200_pw1.00_E50.pt"),
}

# --- Dynamic selection based on CFG.PROBLEM ---
if CFG.PROBLEM == 'AC3D':
    CKPTS = CKPTS_AC3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    IC_FUNCTION = 'create_initial_condition_sphere_ac3d'
    IC_TYPE = 'sphere'
elif CFG.PROBLEM == 'CH3D':
    CKPTS = CKPTS_CH3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    IC_FUNCTION = 'create_initial_condition_sphere_ch3d'
    IC_TYPE = 'sphere'
elif CFG.PROBLEM == 'SH3D':
    CKPTS = CKPTS_SH3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    IC_FUNCTION = 'create_initial_condition_sphere_sh3d'
    IC_TYPE = 'sphere'
elif CFG.PROBLEM == 'PFC3D':
    CKPTS = CKPTS_PFC3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    IC_FUNCTION = 'create_initial_condition_star_pfc3d'
    IC_TYPE = 'star'
elif CFG.PROBLEM == 'MBE3D':
    CKPTS = CKPTS_MBE3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    IC_FUNCTION = 'create_initial_condition_torus_mbe3d'
    IC_TYPE = 'torus'
else:
    raise ValueError(f"Problem '{CFG.PROBLEM}' not configured in this script.")

# -----------------------------------
# 2) Initial Conditions
# -----------------------------------
def _grid(S, L):
    """Helper to create a 3D grid."""
    x = np.linspace(-0.5 * L, 0.5 * L, S, endpoint=False)
    y = z = x
    return np.meshgrid(x, y, z, indexing="ij")


def create_initial_condition_sphere_ac3d():
    """Sphere IC for the AC3D problem (OOD for GRF-trained models)."""
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    radius = 0.5
    xx, yy, zz = _grid(S, L)
    interface_width = np.sqrt(2.0) * epsilon
    u0 = np.tanh((radius - np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)) / interface_width).astype(np.float32)

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

# (keeping the rest of your IC functions unchanged – CH3D/SH3D/PFC3D/MBE3D)
# ... (all the create_initial_condition_* functions you pasted, unchanged) ...

def create_initial_condition_sphere_ch3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    radius = 0.5
    xx, yy, zz = _grid(S, L)
    interface_width = np.sqrt(16.0) * epsilon
    u0 = np.tanh((radius - np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)) / interface_width).astype(np.float32)

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_star_ch3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2) + 1e-9
    theta = np.arccos(np.clip(zz / r, -1.0, 1.0))
    phi = np.arctan2(yy, xx)
    base_r = 0.5
    amp = 0.10
    freq = 6
    mod = 1.0 + amp * (np.cos(freq * theta) * np.cos(freq * phi))
    r_star = base_r * mod
    sdf = r_star - r
    u0 = np.tanh(sdf / (np.sqrt(2) * epsilon)).astype(np.float32)
    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_sphere_sh3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    eps = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    dx = L / S
    x = (-L/2) + dx * np.arange(S, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")

    R = 2.5
    w = max(2.0 * dx, 2.0 * float(eps))
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    u0 = np.tanh((R - r) / w).astype(np.float32)
    u0 = (u0 - u0.mean()).astype(np.float32)
    u0 = np.clip(u0, -0.9, 0.9)
    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_star_pfc3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)
    theta = np.arctan2(zz, xx)
    R_theta = 5.0 + 1.0 * np.cos(6 * theta)
    dist = np.sqrt(xx ** 2 + 2 * yy ** 2 + zz ** 2)
    interface_width = np.sqrt(2) * epsilon
    u0 = np.tanh((R_theta - dist) / interface_width).astype(np.float32)
    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_sphere_pfc3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    R = 6.0
    interface_width = np.sqrt(2.0) * epsilon
    u0 = np.tanh((R - r) / interface_width).astype(np.float32)
    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_sphere_mbe3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)
    R = 0.5
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    phi = R - r
    hx = L / S
    w = max(1.0 * hx, 2.0 * float(epsilon))
    u0 = np.tanh(phi / w).astype(np.float32)
    u0 = u0 - np.float32(u0.mean())
    u0 = np.clip(u0, -0.999, 0.999)
    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_torus_mbe3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)
    R = 0.5
    r0 = 0.1
    rho = np.sqrt(xx**2 + yy**2)
    phi = r0 - np.sqrt((rho - R)**2 + zz**2)
    hx = L / S
    w = max(3.0 * hx, 2.0 * float(epsilon))
    u0 = np.tanh(phi / w).astype(np.float32)
    u0 = u0 - np.float32(u0.mean())
    u0 = np.clip(u0, -0.999, 0.999)
    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_star_mbe3d():
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)
    theta = np.arctan2(zz, xx)
    R0 = 5.0
    A = 1.0
    R_theta = R0 + A * np.cos(6.0 * theta)
    dist = np.sqrt(xx**2 + 2.0 * yy**2 + zz**2)
    w = np.sqrt(2.0) * float(epsilon)
    u0 = np.tanh((R_theta - dist) / w).astype(np.float32)
    u0 = u0 - np.float32(u0.mean())
    u0 = np.clip(u0, -0.999, 0.999)
    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

# ----------------------------------------------------
# 3) Model Loading
# ----------------------------------------------------
def load_model(method: str, model_type: str, device: torch.device):
    path = CKPTS[method][1]
    ckpt = torch.load(path, map_location="cpu")

    if model_type == "FNO4d":
        model = FNO4d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES, modes4_internal=None,
            width=CFG.WIDTH, width_q=getattr(CFG, "WIDTH_Q", CFG.WIDTH),
            T_in_channels=CFG.T_IN_CHANNELS, n_layers=CFG.N_LAYERS
        ).to(device)

    elif model_type == "TNO3d":
        sd = ckpt["state_dict"]
        wq = sd.get("q.layers.0.weight", None)
        wh = sd.get("h.layers.0.weight", None)
        width_q = int(wq.shape[0]) if wq is not None else CFG.WIDTH_Q
        width_h = int(wh.shape[0]) if wh is not None else CFG.WIDTH_H

        model = TNO3d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES,
            width=CFG.WIDTH, width_q=width_q, width_h=width_h,
            T_in=CFG.T_IN_CHANNELS, T_out=1, n_layers=CFG.N_LAYERS
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model

# -----------------------------------------------------------
# 4) Semi-implicit bootstrap for non-AC problems (unchanged)
# -----------------------------------------------------------
import torch.fft as tfft

def semi_implicit_step_ch3d(u, dt, dx, epsilon):
    u = u[..., 0]
    B, Sx, Sy, Sz = u.shape
    kx = 2 * np.pi * np.fft.fftfreq(Sx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Sy, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Sz, d=dx)
    KX2 = torch.from_numpy(kx.astype(np.float32)**2).to(u.device)
    KY2 = torch.from_numpy(ky.astype(np.float32)**2).to(u.device)
    KZ2 = torch.from_numpy(kz.astype(np.float32)**2).to(u.device)
    K2x, K2y, K2z = torch.meshgrid(KX2, KY2, KZ2, indexing="ij")
    K2 = K2x + K2y + K2z
    U = tfft.fftn(u, dim=(-3, -2, -1))
    nonlin = u**3 - 3.0 * u
    NL = tfft.fftn(nonlin, dim=(-3, -2, -1))
    numer = U - dt * K2 * NL
    denom = 1.0 + dt * (2.0 * K2 + (epsilon**2) * (K2**2))
    V = numer / denom
    u_next = tfft.ifftn(V, dim=(-3, -2, -1)).real
    return u_next.unsqueeze(-1)

def semi_implicit_step_sh3d(u, dt, dx, epsilon):
    u = u[..., 0]
    B, Sx, Sy, Sz = u.shape
    kx = 2 * np.pi * np.fft.fftfreq(Sx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Sy, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Sz, d=dx)
    KX2 = torch.from_numpy((kx.astype(np.float32))**2).to(u.device)
    KY2 = torch.from_numpy((ky.astype(np.float32))**2).to(u.device)
    KZ2 = torch.from_numpy((kz.astype(np.float32))**2).to(u.device)
    K2x, K2y, K2z = torch.meshgrid(KX2, KY2, KZ2, indexing="ij")
    K2 = K2x + K2y + K2z
    U  = tfft.fftn(u, dim=(-3, -2, -1))
    NL = tfft.fftn(u**3, dim=(-3, -2, -1))
    numer = (U / dt) - NL + 2.0 * K2 * U
    denom = (1.0 / dt) + (1.0 - float(epsilon)) + (K2 ** 2)
    V = numer / denom
    u_next = tfft.ifftn(V, dim=(-3, -2, -1)).real
    return u_next.unsqueeze(-1)

@torch.no_grad()
def bootstrap_states_from_u0(u0_np: np.ndarray, T_in: int, device: torch.device):
    u = torch.from_numpy(u0_np).to(device=device).unsqueeze(0).unsqueeze(-1)
    states = [u]
    for _ in range(1, T_in):
        if CFG.PROBLEM == 'PFC3D':
            u = semi_implicit_step_pfc(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)
        elif CFG.PROBLEM == 'CH3D':
            u = semi_implicit_step_ch3d(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)
        elif CFG.PROBLEM == 'MBE3D':
            u = semi_implicit_step_mbe(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)
        elif CFG.PROBLEM == 'SH3D':
            u = semi_implicit_step_sh3d(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)
        else:
            u = semi_implicit_step(u, CFG.DT, CFG.DX, CFG.EPS2)  # AC/SH original
        states.append(u)
    x0 = torch.cat(states, dim=-1)
    return states, x0

@torch.no_grad()
def bootstrap_states_from_u0_pfc(u0_np: np.ndarray, T_in: int, device: torch.device):
    u = torch.from_numpy(u0_np).to(device=device).unsqueeze(0).unsqueeze(-1)
    states = [u]
    for _ in range(1, T_in):
        u = semi_implicit_step_pfc(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)
        states.append(u)
    x0 = torch.cat(states, dim=-1)
    return states, x0

# -----------------------------------------------------------
# 5) Rollout autoregressively (original version, used for non-AC)
# -----------------------------------------------------------
@torch.no_grad()
def rollout_aligned(model, x0: torch.Tensor, teacher_states: list, Nt: int):
    T_in = x0.shape[-1]
    seq = [st.squeeze(0).squeeze(-1).detach().cpu().numpy() for st in teacher_states]

    x = x0.clone()
    steps_to_predict = Nt + 1 - T_in
    for _ in range(steps_to_predict):
        y_pred = model(x)
        seq.append(y_pred.squeeze(0).squeeze(-1).cpu().numpy())
        x = torch.cat([x[..., 1:], y_pred], dim=-1)

    return np.stack(seq, axis=0).astype(np.float32)

# -----------------------------------------------------------
# 6) AC3D EXACT SPECTRAL SOLVER + model rollout (torch-only)
# -----------------------------------------------------------
def build_k2_grid_ac3d(nx, ny, nz, dx, device: torch.device):
    Lx = nx * dx
    Ly = ny * dx
    Lz = nz * dx

    def k_vec(N, L):
        k_pos = np.arange(0, N // 2 + 1, dtype=np.float32)
        k_neg = np.arange(-N // 2 + 1, 0, dtype=np.float32)
        k_all = np.concatenate([k_pos, k_neg])
        return (2.0 * np.pi / L) * k_all

    kx = k_vec(nx, Lx)
    ky = k_vec(ny, Ly)
    kz = k_vec(nz, Lz)

    kx2 = torch.tensor(kx**2, device=device)
    ky2 = torch.tensor(ky**2, device=device)
    kz2 = torch.tensor(kz**2, device=device)

    kxx, kyy, kzz = torch.meshgrid(kx2, ky2, kz2, indexing="ij")
    k2 = kxx + kyy + kzz
    return k2

def ac3d_solve_full_trajectory(u0_np, dt, dx, eps2, Nt, device: torch.device, k2=None):
    nx, ny, nz = u0_np.shape
    Cahn = eps2

    if k2 is None:
        k2 = build_k2_grid_ac3d(nx, ny, nz, dx, device)

    u = torch.from_numpy(u0_np.astype(np.float32)).to(device)
    traj = np.zeros((Nt + 1, nx, ny, nz), dtype=np.float32)
    traj[0] = u0_np

    with torch.inference_mode():
        for it in range(1, Nt + 1):
            nonlinear_term_hat = torch.fft.fftn(u**3 - u)
            u_hat = torch.fft.fftn(u)
            v_hat = (u_hat - (dt / Cahn) * nonlinear_term_hat) / (1.0 + dt * k2)
            u = torch.fft.ifftn(v_hat).real
            traj[it] = u.detach().cpu().numpy().astype(np.float32)

    return traj

def time_ac3d_numerical_solver(u0_np, dt, dx, eps2, Nt, device: torch.device, k2=None, n_reps=5):
    nx, ny, nz = u0_np.shape
    Cahn = eps2

    if k2 is None:
        k2 = build_k2_grid_ac3d(nx, ny, nz, dx, device)

    u0 = torch.from_numpy(u0_np.astype(np.float32)).to(device)

    with torch.inference_mode():
        u_tmp = u0.clone()
        for _ in range(min(Nt, 3)):
            nonlinear_term_hat = torch.fft.fftn(u_tmp**3 - u_tmp)
            u_hat = torch.fft.fftn(u_tmp)
            v_hat = (u_hat - (dt / Cahn) * nonlinear_term_hat) / (1.0 + dt * k2)
            u_tmp = torch.fft.ifftn(v_hat).real

    times = []
    for _ in range(n_reps):
        u = u0.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(Nt):
                nonlinear_term_hat = torch.fft.fftn(u**3 - u)
                u_hat = torch.fft.fftn(u)
                v_hat = (u_hat - (dt / Cahn) * nonlinear_term_hat) / (1.0 + dt * k2)
                u = torch.fft.ifftn(v_hat).real
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        times.append(t2 - t1)

    return float(np.mean(times)), float(np.std(times))

def rollout_autoregressive_torch_ac(model, traj_np, T_in, Nt, device: torch.device):
    """
    Torch-only rollout: first T_in frames from exact AC solver,
    subsequent frames predicted autoregressively by the model.
    """
    traj0 = torch.from_numpy(traj_np.astype(np.float32)).to(device)  # (Nt+1,S,S,S)
    total_steps, Sx, Sy, Sz = traj0.shape
    assert total_steps >= Nt + 1
    assert T_in <= Nt

    pred = traj0.clone()

    with torch.inference_mode():
        for t in range(T_in, Nt + 1):
            x = pred[t - T_in:t]      # (T_in,Sx,Sy,Sz)
            x = x.permute(1, 2, 3, 0) # (Sx,Sy,Sz,T_in)
            x = x.unsqueeze(0)        # (1,Sx,Sy,Sz,T_in)
            y_pred = model(x)
            if y_pred.ndim == 5 and y_pred.shape[-1] == 1:
                y_pred = y_pred[..., 0]
            y_pred = y_pred[0]
            pred[t] = y_pred

    return pred.detach().cpu().numpy().astype(np.float32)

def time_model_rollout_ac(model, traj_np, T_in, Nt, device: torch.device, n_reps=5):
    _ = rollout_autoregressive_torch_ac(model, traj_np, T_in, Nt, device)
    times = []
    for _ in range(n_reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = rollout_autoregressive_torch_ac(model, traj_np, T_in, Nt, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        times.append(t2 - t1)
    return float(np.mean(times)), float(np.std(times))

# -----------------------------------------------------------
# 7) High-quality 3D plotting
# -----------------------------------------------------------
def plot_isosurface_grid(volumes_by_method, method_order, frames, out_png, out_pdf, upsample=2, pad_vox=2):
    n_rows = len(method_order)
    n_cols = len(frames)

    fig = plt.figure(figsize=(4.8 * n_cols, 4.2 * n_rows))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.05, wspace=0.05, hspace=0.05)

    facecolor_map = {
        "FNO4d": (1.00, 0.10, 0.10, 1.0),
        "MHNO": (0.95, 0.20, 0.20, 1.0),
        "PENCO-MHNO": (0.10, 0.90, 0.10, 1.0),
        "PENCO-FNO": (0.95, 0.50, 0.15, 1.0),
        "PENCO": (0.10, 0.90, 0.10, 1.0),
        "PurePhysics": (0.05, 0.95, 0.05, 1.0),
        "Exact": (0.20, 0.20, 0.90, 1.0),
        "Solver": (0.20, 0.20, 0.90, 1.0),
    }

    def _tighten_axes(ax, verts, pad=pad_vox):
        if verts.size == 0:
            return
        mins, maxs = verts.min(axis=0), verts.max(axis=0)
        ax.set_xlim(mins[2] - pad, maxs[2] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        ax.set_zlim(mins[0] - pad, maxs[0] + pad)

    plot_idx = 1
    for r, method in enumerate(method_order):
        vols = volumes_by_method[method]
        for c, t in enumerate(frames):
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            plot_idx += 1
            ax.set_facecolor('white')
            ax.grid(False)

            vol_zyx = np.transpose(vols[t], (2, 1, 0))
            if upsample > 1:
                vol_zyx = nd_zoom(vol_zyx, zoom=upsample, order=1)

            try:
                verts, faces, _, _ = marching_cubes(vol_zyx, level=0.0)
                rgba = facecolor_map.get(method, (0.5, 0.5, 0.5, 1.0))
                mesh = Poly3DCollection(verts[faces], alpha=rgba[3], facecolor=rgba, edgecolor='none')
                ax.add_collection3d(mesh)
                _tighten_axes(ax, verts, pad=pad_vox)
            except Exception as e:
                ax.text2D(0.05, 0.90, f"MC fail @ t={t}\n{e}", transform=ax.transAxes)

            if r == 0:
                ax.set_title(rf"${t}\,\Delta t$", fontsize=20, fontweight='bold', pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=-60)

        fig.text(0.03, 1.0 - (r + 0.5) / n_rows, method, va='center', ha='center',
                 rotation=90, fontsize=22, fontweight='bold')

    plt.savefig(out_png, dpi=400, facecolor='white')
    plt.savefig(out_pdf, dpi=400, facecolor='white')
    plt.close(fig)
    print(f"Saved comparison figures to: {out_png} and {out_pdf}")

# -----------------------------------------------------------
# 8) Main execution block
# -----------------------------------------------------------
def main():
    device = CFG.DEVICE if isinstance(CFG.DEVICE, torch.device) else torch.device(CFG.DEVICE)

    ic_func = globals()[IC_FUNCTION]
    u0_np, domain_lengths, grid_sizes, Nt, dt, selected_frames = ic_func()

    print(f"[IC diagnostics]")
    print(f"  u0 range: [{u0_np.min():.3f}, {u0_np.max():.3f}]")
    print(f"  u0 mean: {u0_np.mean():.3f}")
    print(f"  Grid: {grid_sizes}, Domain: {domain_lengths}")
    print(f"  epsilon_param: {CFG.EPSILON_PARAM}, dt: {dt}, L_DOMAIN: {CFG.L_DOMAIN}")

    Sx, Sy, Sz = grid_sizes
    assert (Sx, Sy, Sz) == (CFG.GRID_RESOLUTION,) * 3, \
        f"Grid mismatch: IC {grid_sizes} vs CFG {CFG.GRID_RESOLUTION}"
    assert abs(dt - CFG.DT) < 1e-12, "DT mismatch between IC and CFG"

    volumes_by_method = {}
    timing_results = {}

    # =========================
    # AC3D: use exact spectral solver + timings
    # =========================
    if CFG.PROBLEM == 'AC3D':
        print("\n==== AC3D: exact spectral solver + OOD sphere IC ====")
        nx = ny = nz = CFG.GRID_RESOLUTION
        dx = CFG.DX
        eps2 = CFG.EPS2
        T_in = CFG.T_IN_CHANNELS

        # Build k^2 grid
        k2 = build_k2_grid_ac3d(nx, ny, nz, dx, device)

        # Exact AC trajectory
        traj_exact = ac3d_solve_full_trajectory(u0_np, CFG.DT, CFG.DX, eps2, Nt, device, k2=k2)
        volumes_by_method["Exact"] = traj_exact  # if you want to plot or inspect later

        # Solver runtime
        solver_mean, solver_std = time_ac3d_numerical_solver(
            u0_np, CFG.DT, CFG.DX, eps2, Nt, device, k2=k2, n_reps=N_REPS_TIME
        )
        timing_results["Solver"] = (solver_mean, solver_std)

        print("\n===== AC3D Numerical Solver (reference) =====")
        print(f"AC3D numerical solver: {solver_mean:.4f} s ± {solver_std:.4f} s\n")

        # Models: rollout + timings (OOD sphere IC)
        for method in METHODS:
            print(f"--- Benchmarking model: {method} ---")
            model_type, _ = CKPTS[method]
            model = load_model(method, model_type, device=device)

            mean_model, std_model = time_model_rollout_ac(
                model, traj_exact, T_in, Nt, device, n_reps=N_REPS_TIME
            )

            speedup = solver_mean / mean_model if mean_model > 0 else float("nan")
            timing_results[method] = (mean_model, std_model, speedup)

            print(f"Model rollout ({method}): {mean_model:.4f} s ± {std_model:.4f} s")
            print(f"Speedup (solver / {method}): {speedup:.3f}×\n")

            # Full trajectory: exact for first T_in, then model predictions
            vols = rollout_autoregressive_torch_ac(model, traj_exact, T_in, Nt, device)
            volumes_by_method[method] = vols

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("===== Summary: model vs AC3D solver (same IC, same Nt, same grid, same device) =====")
        print(f"AC3D solver: {solver_mean:.4f} s ± {solver_std:.4f} s")
        for method in METHODS:
            m_mean, m_std, sp = timing_results[method]
            print(
                f"{method:12s}  model: {m_mean:.4f} s ± {m_std:.4f} s   "
                f"speedup (solver/model): {sp:.3f}×"
            )

        # Teacher window for .mat: first T_in exact frames
        teacher_window = traj_exact[:CFG.T_IN_CHANNELS]

    # =========================
    # Other problems: keep old behavior (no exact solver timing)
    # =========================
    else:
        print(f"\n==== {CFG.PROBLEM}: using semi-implicit teacher as before ====")
        if CFG.PROBLEM == 'PFC3D':
            teacher_states, x0 = bootstrap_states_from_u0_pfc(u0_np, CFG.T_IN_CHANNELS, device=device)
        else:
            teacher_states, x0 = bootstrap_states_from_u0(u0_np, CFG.T_IN_CHANNELS, device=device)

        def _check_finite(vols, name):
            vmin, vmax = np.nanmin(vols), np.nanmax(vols)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax > 10 or vmin < -10:
                print(f"[WARN] {name}: suspicious values: min={vmin:.3g}, max={vmax:.3g}")
            return vols

        for method in METHODS:
            print(f"Loading & rolling out: {method}")
            model_type, _ = CKPTS[method]
            model = load_model(method, model_type, device=device)
            vols = rollout_aligned(model, x0, teacher_states, Nt=Nt)
            vols = _check_finite(vols, method)
            volumes_by_method[method] = vols
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        teacher_window = np.stack([s.squeeze().cpu().numpy() for s in teacher_states], axis=0)

    # ------------------
    # Visualization (models only by default)
    # ------------------
    base_filename = f"{CFG.PROBLEM.lower()}_{IC_TYPE}_methods_compare"
    out_png = f"{base_filename}.png"
    out_pdf = f"{base_filename}.pdf"

    plot_methods = METHODS  # if you want to also plot 'Exact', append it here

    plot_isosurface_grid(
        volumes_by_method, plot_methods, selected_frames,
        out_png=out_png,
        out_pdf=out_pdf,
        upsample=3, pad_vox=3
    )

    # ------------------
    # Save .mat file
    # ------------------
    out_data = {
        'meta': {
            'case': f'{CFG.PROBLEM}_{IC_TYPE}',
            'grid_sizes': np.array(grid_sizes, dtype=np.int32),
            'domain_lengths': np.array(domain_lengths, dtype=np.float32),
            'dx': np.float32(CFG.DX),
            'dt': np.float32(dt),
            'Nt': np.int32(Nt),
            'selected_frames': np.array(selected_frames, dtype=np.int32),
        },
        'U0': u0_np,
        'teacher_window': teacher_window,
    }

    if CFG.PROBLEM == 'AC3D':
        out_data['Exact'] = volumes_by_method["Exact"]
        # add timing info for solver + models
        out_data['runtime'] = {
            'solver_mean': timing_results["Solver"][0],
            'solver_std':  timing_results["Solver"][1],
            'methods': {m: timing_results[m] for m in METHODS},
        }

    for method in METHODS:
        matlab_name = method.replace('-', '_')
        out_data[matlab_name] = volumes_by_method[method]

    mat_name = f"{CFG.PROBLEM.lower()}_{IC_TYPE}_eval_compare.mat"
    savemat(mat_name, out_data, do_compression=True)
    print(f"Saved MATLAB results to: {mat_name}")


if __name__ == "__main__":
    main()
