import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.io import savemat
from scipy.ndimage import zoom as nd_zoom

# --- repo imports (your files) ---
import config as CFG
from networks import FNO4d, TNO3d
from functions import semi_implicit_step, semi_implicit_step_pfc, semi_implicit_step_mbe, semi_implicit_step_sh, mass_project_pred
import time  # NEW: for timing

# -------------------------------------------------------------------
# 1) PROBLEM-SPECIFIC CONFIGURATIONS
#    - Selects checkpoints, methods, and initial conditions
#      based on the PROBLEM variable in config.py
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
    # NEW: split PENCO into two variants for CH3D (as in AC3D)
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
              "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/FNO4d_FNO4d_N200_pw0.00_E150.pt"),
    "MHNO": ("TNO3d",
             "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/TNO3d_MHNO_N200_pw0.00_E150.pt"),
    # NEW: split PENCO into two variants for CH3D (as in AC3D)
    "PENCO-MHNO": ("TNO3d",
                   "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/TNO3d_PENCO_N200_pw0.25_E150.pt"),
    "PENCO-FNO": ("FNO4d",
                  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/FNO4d_PENCO_N200_pw0.25_E150.pt"),
    "PurePhysics": ("TNO3d",
                    "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/SH3d_models/TNO3d_PurePhysics_N200_pw1.00_E150.pt"),
}

# --- Checkpoints for PFC3D ---
CKPTS_PFC3D = {
    # method_label: (model_type, path)
    "FNO4d": ("FNO4d",
              "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/FNO4d_FNO4d_N200_pw0.00_E50.pt"),
    "MHNO": ("TNO3d",
             "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/TNO3d_MHNO_N200_pw0.00_E50.pt"),
    # NEW: split PENCO into two variants for CH3D (as in AC3D)
    "PENCO-MHNO": ("TNO3d",
                   "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/TNO3d_PENCO_N200_pw0.25_E50.pt"),
    "PENCO-FNO": ("FNO4d",
                  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/PFC3d_models/FNO4d_PENCO_N200_pw0.25_E50.pt"),
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
    #IC_FUNCTION = 'create_initial_condition_star_ch3d'
    #IC_FUNCTION = 'create_initial_condition_spinodal_ch3d'
    IC_TYPE = 'sphere' # sphere, star

elif CFG.PROBLEM == 'SH3D':
    CKPTS = CKPTS_SH3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    IC_FUNCTION = 'create_initial_condition_sphere_sh3d'
    IC_TYPE = 'sphere'
elif CFG.PROBLEM == 'PFC3D':
    CKPTS = CKPTS_PFC3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]  # <- now 5 like AC3D
    IC_FUNCTION = 'create_initial_condition_star_pfc3d'
    #IC_FUNCTION ='create_initial_condition_sphere_pfc3d'
    IC_TYPE = 'star'
elif CFG.PROBLEM == 'MBE3D':
    CKPTS = CKPTS_MBE3D
    METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    IC_FUNCTION = 'create_initial_condition_torus_mbe3d'
    #IC_FUNCTION ='create_initial_condition_sphere_mbe3d'
    IC_TYPE = 'torus' # torus   sphere
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
    """Generates the sphere IC for the AC3D problem."""
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    radius = 0.5 # 0.5
    xx, yy, zz = _grid(S, L)
    interface_width = np.sqrt(2.0) * epsilon
    u0 = np.tanh((radius - np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)) / interface_width).astype(np.float32)

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_sphere_ch3d():
    """Generates the sphere IC for the AC3D problem."""
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    radius = 0.5 # 0.5
    xx, yy, zz = _grid(S, L)
    interface_width = np.sqrt(2.0) * epsilon
    u0 = np.tanh((radius - np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)) / interface_width).astype(np.float32)

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames


def create_initial_condition_sphere_sh3d():
    """
    Balanced (zero-mean) sphere IC for SH3D to avoid saturation at -1.
    This makes the IC comparable to others and plots look sensible.
    """
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    eps = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    # grid (half-open to match numpy endpoint=False style)
    dx = L / S
    x = (-L/2) + dx * np.arange(S, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")

    # make the sphere bigger and the interface not razor-thin
    R = 5.0 # 2.5                               # was 1.5 → tiny sphere → mean ~ -1
    w = max(2.0 * dx, 2.0 * float(eps))   # thicker interface

    r = np.sqrt(xx**2 + yy**2 + zz**2)
    u0 = np.tanh((R - r) / w).astype(np.float32)

    # balance it
    u0 = (u0 - u0.mean()).astype(np.float32)  # zero-mean
    u0 = np.clip(u0, -0.9, 0.9)               # avoid extremes

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames


def create_initial_condition_star_pfc3d():
    """Generates the star IC for the PFC3D problem (SMOOTH version to match MATLAB)."""
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)

    # Star shape parameters (EXACTLY as in MATLAB)
    theta = np.arctan2(zz, xx)
    R_theta = 10.0 + 5.0 * np.cos(6 * theta)
    dist = np.sqrt(xx ** 2 + 2 * yy ** 2 + zz ** 2)

    # SMOOTH initial condition (matches MATLAB stable version)
    interface_width = np.sqrt(49) * epsilon
    u0 = np.tanh((R_theta - dist) / interface_width).astype(np.float32)  # <-- SMOOTH!


    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames


def create_initial_condition_sphere_pfc3d():
    """
    Generates the SPHERE IC for the PFC3D problem (unseen input for models).
    Form: u0 = tanh((R - r) / (sqrt(2)*epsilon))  -- same as our MATLAB PFC sphere.
    """
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    xx, yy, zz = _grid(S, L)
    r = np.sqrt(xx**2 + yy**2 + zz**2)

    # Radius choice to mirror the MATLAB PFC3D example on L=10*pi
    # (if your config uses a different L, feel free to adjust R)
    R = 5.0
    interface_width = np.sqrt(2.0) * epsilon
    u0 = np.tanh((R - r) / interface_width).astype(np.float32)

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_torus_mbe3d():
    """
    Smooth tanh torus IC for MBE3D, consistent with the dataset:
      - Domain: [-L/2, L/2]^3 with S^3 grid
      - Major radius R ≈ 1.9, tube radius r0 ≈ 1.0
      - Interface thickness w ≈ max(3*h, 2*epsilon)
      - Zero-mean and gentle clamp to (-0.999, 0.999)
    """
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    # grid
    xx, yy, zz = _grid(S, L)

    # torus geometry (inside ~[-π, π]^3 like your MATLAB code)
    R = 0.5       # major radius
    r0 = 0.1      # tube radius
    rho = np.sqrt(xx**2 + yy**2)
    phi = r0 - np.sqrt((rho - R)**2 + zz**2)  # signed distance–like

    # interface width: match MATLAB/description
    hx = L / S
    w = max(3.0 * hx, 2.0 * float(epsilon))

    u0 = np.tanh(phi / w).astype(np.float32)

    # preprocessing: zero mean, gentle clamp
    u0 = u0 - np.float32(u0.mean())
    u0 = np.clip(u0, -0.999, 0.999)

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

def create_initial_condition_star_mbe3d():
    """
    Star-shaped IC analogous to your MATLAB 'star' case:

        Nx = Ny = Nz = 32
        Lx = 10*pi (bigger domain)            <-- see note below
        epsilon = 0.5
        theta = atan2(zz, xx)
        R_theta = 5.0 + 1.0*cos(6*theta)
        dist = sqrt(xx^2 + 2*yy^2 + zz^2)
        u = tanh((R_theta - dist)/sqrt(2)*epsilon)

    Here we keep the same *style* as your other Python ICs
    (use CFG, zero-mean, clamp). For *exact* geometric match
    to your MATLAB star, set:

        CFG.L_DOMAIN     = 10*np.pi
        CFG.EPSILON_PARAM = 0.5
    """
    S = CFG.GRID_RESOLUTION
    L = CFG.L_DOMAIN          # set this to 10*np.pi to match MATLAB exactly
    epsilon = CFG.EPSILON_PARAM
    dt = float(CFG.DT)
    Nt = CFG.TOTAL_TIME_STEPS
    selected_frames = [0, 20, 40, 60, 80, 100]

    # grid
    xx, yy, zz = _grid(S, L)

    # star-shaped geometry
    theta = np.arctan2(zz, xx)           # angle in x–z plane
    R0 = 5.0
    A = 1.0
    R_theta = R0 + A * np.cos(6.0 * theta)  # 6-pointed star

    dist = np.sqrt(xx**2 + 2.0 * yy**2 + zz**2)

    # interface width (from MATLAB: sqrt(2)*epsilon)
    w = np.sqrt(2.0) * float(epsilon)

    u0 = np.tanh((R_theta - dist) / w).astype(np.float32)

    # make it consistent with your other ICs: zero-mean, clamp
    u0 = u0 - np.float32(u0.mean())
    u0 = np.clip(u0, -0.999, 0.999)

    return u0, (L, L, L), (S, S, S), Nt, dt, selected_frames

# ----------------------------------------------------
# 3) Model Loading
# ----------------------------------------------------
def load_model(method: str, model_type: str, device: torch.device):
    path = CKPTS[method][1]
    # Avoid weights_only to keep broad compatibility
    ckpt = torch.load(path, map_location="cpu")

    if model_type == "FNO4d":
        model = FNO4d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES, modes4_internal=None,
            width=CFG.WIDTH, width_q=getattr(CFG, "WIDTH_Q", CFG.WIDTH),  # harmless
            T_in_channels=CFG.T_IN_CHANNELS, n_layers=CFG.N_LAYERS
        ).to(device)

    elif model_type == "TNO3d":
        sd = ckpt["state_dict"]

        # Infer widths from the checkpoint if present; otherwise fall back to CFG
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
    # Move to device after loading for safety if you prefer:
    model.to(device)
    return model

## CH3D
import torch.fft as tfft

def semi_implicit_step_ch3d(u, dt, dx, epsilon):
    """
    CH3D semi-implicit (matches MATLAB):
        u^{n+1} = ifftn( (Û - dt*K2*F[u^3 - 3u]) / (1 + dt*(2*K2 + eps^2*K2^2)) )
    u: (B,S,S,S,1), real
    """
    # squeeze channel
    u = u[..., 0]  # (B,Sx,Sy,Sz)

    B, Sx, Sy, Sz = u.shape
    # Angular wavenumbers (rad/unit length). IMPORTANT: NO extra /L.
    # np.fft.fftfreq(n, d=dx) gives cycles per unit; multiply by 2π to get rad/unit.
    kx = 2 * np.pi * np.fft.fftfreq(Sx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Sy, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Sz, d=dx)

    # Build K^2 grid on the correct device/dtype
    KX2 = torch.from_numpy(kx.astype(np.float32)**2).to(u.device)
    KY2 = torch.from_numpy(ky.astype(np.float32)**2).to(u.device)
    KZ2 = torch.from_numpy(kz.astype(np.float32)**2).to(u.device)
    K2x, K2y, K2z = torch.meshgrid(KX2, KY2, KZ2, indexing="ij")
    K2 = K2x + K2y + K2z
    del K2x, K2y, K2z

    U = tfft.fftn(u, dim=(-3, -2, -1))
    nonlin = u**3 - 3.0 * u
    NL = tfft.fftn(nonlin, dim=(-3, -2, -1))

    numer = U - dt * K2 * NL
    denom = 1.0 + dt * (2.0 * K2 + (epsilon**2) * (K2**2))

    V = numer / denom
    u_next = tfft.ifftn(V, dim=(-3, -2, -1)).real  # (B,Sx,Sy,Sz)
    return u_next.unsqueeze(-1)                     # (B,Sx,Sy,Sz,1)

import torch.fft as tfft

def semi_implicit_step_sh3d(u, dt, dx, epsilon):
    """
    SH3D semi-implicit (matches your MATLAB DNS):
        v̂ = (Û/dt - F[u^3] + 2*K2*Û) / (1/dt + (1-ε) + K2^2)
        u^{n+1} = ifftn(v̂)
    u: (B,S,S,S,1) real
    """
    u = u[..., 0]  # (B,Sx,Sy,Sz)

    B, Sx, Sy, Sz = u.shape
    kx = 2 * np.pi * np.fft.fftfreq(Sx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Sy, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Sz, d=dx)

    KX2 = torch.from_numpy((kx.astype(np.float32))**2).to(u.device)
    KY2 = torch.from_numpy((ky.astype(np.float32))**2).to(u.device)
    KZ2 = torch.from_numpy((kz.astype(np.float32))**2).to(u.device)
    K2x, K2y, K2z = torch.meshgrid(KX2, KY2, KZ2, indexing="ij")
    K2 = K2x + K2y + K2z
    del K2x, K2y, K2z

    U  = tfft.fftn(u, dim=(-3, -2, -1))
    NL = tfft.fftn(u**3, dim=(-3, -2, -1))

    numer = (U / dt) - NL + 2.0 * K2 * U
    denom = (1.0 / dt) + (1.0 - float(epsilon)) + (K2 ** 2)

    V = numer / denom
    u_next = tfft.ifftn(V, dim=(-3, -2, -1)).real
    return u_next.unsqueeze(-1)

# -----------------------------------------------------------
# 4) Bootstrap T_in window with semi-implicit teacher
# -----------------------------------------------------------
@torch.no_grad()
def bootstrap_states_from_u0(u0_np: np.ndarray, T_in: int, device: torch.device):
    u = torch.from_numpy(u0_np).to(device=device).unsqueeze(0).unsqueeze(-1)
    states = [u]
    for _ in range(1, T_in):
        if CFG.PROBLEM == 'PFC3D':
            u = semi_implicit_step_pfc(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)
        elif CFG.PROBLEM == 'CH3D':
            u = semi_implicit_step_ch3d(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)  # <<< use CH step
        elif CFG.PROBLEM == 'MBE3D':
            u = semi_implicit_step_mbe(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)   # <<< MBE teacher
        elif CFG.PROBLEM == 'SH3D':
            u = semi_implicit_step_sh(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)  # <<< NEW
        else:
            u = semi_implicit_step(u, CFG.DT, CFG.DX, CFG.EPS2)  # AC/SH path as before
        states.append(u)
    x0 = torch.cat(states, dim=-1)
    return states, x0

@torch.no_grad()
def bootstrap_states_from_u0_pfc(u0_np: np.ndarray, T_in: int, device: torch.device):
    """
    Semi-implicit spectral step that matches the MATLAB PFC DNS and training generator:
    v̂ = (Û/dt - K·F[u^3] + 2K^2·Û) / (1/dt + (1-ε)K + K^3),  u^{n+1}=ifftn(v̂)
    """
    # u: (1,S,S,S,1)
    u = torch.from_numpy(u0_np).to(device=device).unsqueeze(0).unsqueeze(-1)
    states = [u]
    for _ in range(1, T_in):
        u = semi_implicit_step_pfc(u, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)  # PFC step!
        states.append(u)
    x0 = torch.cat(states, dim=-1)
    return states, x0


# -----------------------------------------------------------
# 5) Rollout autoregressively
# -----------------------------------------------------------
@torch.no_grad()
def rollout_aligned(model, x0: torch.Tensor, teacher_states: list, Nt: int):
    T_in = x0.shape[-1]
    seq = [st.squeeze(0).squeeze(-1).detach().cpu().numpy() for st in teacher_states]

    x = x0.clone()
    steps_to_predict = Nt + 1 - T_in
    for _ in range(steps_to_predict):
        y_pred = model(x)

        # --- NEW: enforce CH mass conservation just like in training ---
        #if CFG.PROBLEM == 'CH3D':
        #    last_in = x[..., -1:]  # (B,S,S,S,1)
        #    y_pred = mass_project_pred(y_raw, last_in)

        #elif CFG.PROBLEM == 'PFC3D':
        #    # <<< NEW: same as rollout_autoregressive >>>
        #    last_in = x[..., -1:]  # (B,S,S,S,1)
        #    ##y_pred = physics_guided_update_pfc_optimal(last_in, y_raw, alpha_cap=0.6, low_k_snap_frac=0.45)
        #    y_pred = mass_project_pred(y_raw, last_in)

        #if CFG.PROBLEM == 'MBE3D':
        #    # <<< NEW: same as rollout_autoregressive >>>
        #    last_in = x[..., -1:]  # (B,S,S,S,1)
        #    ##y_pred = physics_guided_update_pfc_optimal(last_in, y_raw, alpha_cap=0.6, low_k_snap_frac=0.45)
        #    y_pred = mass_project_pred(y_raw, last_in)

        #else:
        #    y_pred = y_raw


        seq.append(y_pred.squeeze(0).squeeze(-1).cpu().numpy())
        x = torch.cat([x[..., 1:], y_pred], dim=-1)

    return np.stack(seq, axis=0).astype(np.float32)

# -----------------------------------------------------------
# 6) High-quality 3D plotting
# -----------------------------------------------------------
def plot_isosurface_grid(volumes_by_method, method_order, frames, out_png, out_pdf, upsample=2, pad_vox=2):
    n_rows = len(method_order)
    n_cols = len(frames)

    fig = plt.figure(figsize=(4.8 * n_cols, 4.2 * n_rows))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.05, wspace=0.05, hspace=0.05)

    # Colors
    facecolor_map = {
        "FNO4d": (1.00, 0.10, 0.10, 1.0),
        "MHNO": (0.95, 0.20, 0.20, 1.0),
        "PENCO-MHNO": (0.10, 0.90, 0.10, 1.0),
        "PENCO-FNO": (0.95, 0.50, 0.15, 1.0),
        "PENCO": (0.10, 0.90, 0.10, 1.0),
        "PurePhysics": (0.05, 0.95, 0.05, 1.0),
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
# 7) Main execution block
# -----------------------------------------------------------
def main():
    device = CFG.DEVICE

    # Get the correct IC function from its name and call it
    ic_func = globals()[IC_FUNCTION]
    u0_np, domain_lengths, grid_sizes, Nt, dt, selected_frames = ic_func()

    Sx, Sy, Sz = grid_sizes
    assert (Sx, Sy, Sz) == (CFG.GRID_RESOLUTION,) * 3, \
        f"Grid mismatch: IC {grid_sizes} vs CFG {CFG.GRID_RESOLUTION}"
    assert abs(dt - CFG.DT) < 1e-12, "DT mismatch between IC and CFG"

    # Build teacher window
    if CFG.PROBLEM == 'PFC3D':
        teacher_states, x0 = bootstrap_states_from_u0_pfc(u0_np, CFG.T_IN_CHANNELS, device=device)
    else:
        teacher_states, x0 = bootstrap_states_from_u0(u0_np, CFG.T_IN_CHANNELS, device=device)

    def _check_finite(vols, name):
        vmin, vmax = np.nanmin(vols), np.nanmax(vols)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax > 10 or vmin < -10:
            print(f"[WARN] {name}: suspicious values: min={vmin:.3g}, max={vmax:.3g}")
        return vols

    # Load models & rollout for the selected problem
    volumes_by_method = {}
    for method in METHODS:
        print(f"Loading & rolling out: {method}")
        model_type, _ = CKPTS[method]
        model = load_model(method, model_type, device=device)
        vols = rollout_aligned(model, x0, teacher_states, Nt=Nt)
        vols = _check_finite(vols, method)  # add this
        # -------------------------------------------------------
        # AFTER YOU COMPUTE 'vols' FOR THIS METHOD
        # -------------------------------------------------------

        if method == "PurePhysics":
            with torch.no_grad():
                u_last = teacher_states[-1]  # (1,32,32,32,1)
                y_raw = model(x0)
                if CFG.PROBLEM == 'CH3D' or CFG.PROBLEM == 'PFC3D' or CFG.PROBLEM == 'MBE3D':
                    y_proj = mass_project_pred(y_raw, u_last)
                    y_show = y_proj
                else:
                    y_show = y_raw
                y_np = y_show.squeeze().cpu().numpy()

            print("PurePhysics single-step debug (after mass projection):")
            print("  teacher last frame: range [%.4f, %.4f], mean=%.4f" %
                  (u_last.min().item(), u_last.max().item(), u_last.mean().item()))
            print("  model next frame:   range [%.4f, %.4f], mean=%.4f" %
                  (y_np.min(), y_np.max(), y_np.mean()))

        volumes_by_method[method] = vols

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate dynamic output filenames
    base_filename = f"{CFG.PROBLEM.lower()}_{IC_TYPE}_methods_compare"
    out_png = f"{base_filename}.png"
    out_pdf = f"{base_filename}.pdf"

    # Visualization
    plot_isosurface_grid(
        volumes_by_method, METHODS, selected_frames,
        out_png=out_png,
        out_pdf=out_pdf,
        upsample=3, pad_vox=3
    )

    # --- Save .mat file ---
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
        'teacher_window': np.stack([s.squeeze().cpu().numpy() for s in teacher_states], axis=0),
        'U0': u0_np,
    }
    # Save each method (Matlab-friendly keys)
    for method in METHODS:
        matlab_friendly_name = method.replace('-', '_')  # e.g., PENCO-MHNO -> PENCO_MHNO
        out_data[matlab_friendly_name] = volumes_by_method[method]

    mat_name = f"{CFG.PROBLEM.lower()}_{IC_TYPE}_eval_compare.mat"
    savemat(mat_name, out_data, do_compression=True)
    print(f"Saved MATLAB results to: {mat_name}")

if __name__ == "__main__":
    device = CFG.DEVICE

    # Get the correct IC function from its name and call it
    ic_func = globals()[IC_FUNCTION]
    u0_np, domain_lengths, grid_sizes, Nt, dt, selected_frames = ic_func()

    # ==================== DIAGNOSTIC CODE ====================
    print(f"Python IC diagnostics:")
    print(f"  u0 range: [{u0_np.min():.3f}, {u0_np.max():.3f}]")
    print(f"  u0 mean: {u0_np.mean():.3f}")
    print(f"  Grid: {grid_sizes}, Domain: {domain_lengths}")
    print(f"  epsilon: {CFG.EPSILON_PARAM}, dt: {dt}")
    print(f"  L_DOMAIN: {CFG.L_DOMAIN}")
    # =========================================================

    Sx, Sy, Sz = grid_sizes
    assert (Sx, Sy, Sz) == (CFG.GRID_RESOLUTION,) * 3, \
        f"Grid mismatch: IC {grid_sizes} vs CFG {CFG.GRID_RESOLUTION}"
    assert abs(dt - CFG.DT) < 1e-12, "DT mismatch between IC and CFG"

    # Build teacher window
    if CFG.PROBLEM == 'PFC3D':
        teacher_states, x0 = bootstrap_states_from_u0_pfc(u0_np, CFG.T_IN_CHANNELS, device=device)
    else:
        teacher_states, x0 = bootstrap_states_from_u0(u0_np, CFG.T_IN_CHANNELS, device=device)

    main()
