import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# --- your repo pieces ---
import config as CFG
from networks import FNO4d, TNO3d
from utilities import rollout_autoregressive
from functions import semi_implicit_step

# -----------------------------
# 1) Where to load checkpoints
# -----------------------------
CKPTS = {
    # method_label: (model_type, path)
    "FNO4d": (
        "FNO4d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/FNO4d_FNO4d_N200_pw0.00_E50.pt",
    ),
    "MHNO": (
        "TNO3d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/TNO3d_MHNO_N200_pw0.00_E50.pt",
    ),
    "PENCO": (
        "TNO3d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/TNO3d_PENCO_N200_pw0.25_E50.pt",
    ),
    "PurePhysics": (
        "TNO3d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/TNO3d_PurePhysics_N200_pw1.00_E50.pt",
    ),
}

# -----------------------------
# 2) OOD initial conditions
# -----------------------------
def _grid(S, L):
    # match MATLAB/config: domain length L, centered at 0, step DX=L/S
    x = np.linspace(-0.5 * L + CFG.DX, 0.5 * L, S, endpoint=True)
    y = x.copy()
    z = x.copy()
    return np.meshgrid(x, y, z, indexing="ij")

def ic_sphere(S=CFG.GRID_RESOLUTION, L=CFG.L_DOMAIN, radius=0.35, eps=CFG.EPSILON_PARAM):
    xx, yy, zz = _grid(S, L)
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    # Smooth double-well profile via tanh of signed distance
    phi = np.tanh((radius - r) / (np.sqrt(2) * eps))
    return phi.astype(np.float32)

def ic_star(S=CFG.GRID_RESOLUTION, L=CFG.L_DOMAIN, base_r=0.25, amp=0.10, freq=5, eps=CFG.EPSILON_PARAM):
    xx, yy, zz = _grid(S, L)
    r = np.sqrt(xx**2 + yy**2 + zz**2) + 1e-8
    # radial “star” modulation with spherical harmonics-like bumps
    theta = np.arccos(np.clip(zz / r, -1.0, 1.0))        # polar
    phi   = np.arctan2(yy, xx)                           # azimuth
    mod   = 1.0 + amp * (np.cos(freq * theta) * np.cos(freq * phi))
    r_star = base_r * mod
    sdf = r_star - r
    u = np.tanh(sdf / (np.sqrt(2) * eps))
    return u.astype(np.float32)

# -----------------------------
# 3) Seed the input window T_in
# -----------------------------
@torch.no_grad()
def seed_window_from_ic(u0_np, T_in=CFG.T_IN_CHANNELS, device=CFG.DEVICE):
    """
    Build the (S,S,S,T_in) window from a single OOD IC by rolling
    the semi-implicit teacher for T_in-1 steps.
    """
    u = torch.from_numpy(u0_np)[None, ...].to(device).unsqueeze(-1)  # (1,S,S,S,1)
    frames = [u.clone()]
    for _ in range(T_in - 1):
        u = semi_implicit_step(u, CFG.DT, CFG.DX, CFG.EPS2)  # (1,S,S,S,1)
        frames.append(u)
    x = torch.cat(frames, dim=-1)  # (1,S,S,S,T_in)
    return x

# -----------------------------
# 4) Load models
# -----------------------------
def build_model(model_type: str):
    if model_type == "FNO4d":
        return FNO4d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES,
            modes4_internal=None, width=CFG.WIDTH, width_q=CFG.WIDTH_Q,
            T_in_channels=CFG.T_IN_CHANNELS, n_layers=CFG.N_LAYERS
        ).to(CFG.DEVICE)
    else:
        return TNO3d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES,
            width=CFG.WIDTH, width_q=CFG.WIDTH_Q, width_h=CFG.WIDTH_H,
            T_in=CFG.T_IN_CHANNELS, T_out=1, n_layers=CFG.N_LAYERS
        ).to(CFG.DEVICE)

@torch.no_grad()
def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# -----------------------------
# 5) Autoregressive rollout on OOD
# -----------------------------
@torch.no_grad()
def rollout_from_window(model, x0, Nt=CFG.TOTAL_TIME_STEPS):
    """
    x0: (1,S,S,S,T_in) tensor (already on device)
    returns (Nt+1, S,S,S) prediction with x0[..., -1] as t=0
    """
    # utilities.rollout_autoregressive expects a full GT trajectory typically,
    # so we do a manual unrolling using the model's one-step mapping:
    preds = []
    # take the last frame in window as t0 “state”
    x_win = x0.clone()
    S = x0.shape[1]
    # First predicted step (t=1)
    y = model(x_win)  # (1,S,S,S,1)
    u = y
    preds.append(x_win[..., -1].squeeze(0).squeeze(-1).cpu().numpy())  # t=0
    preds.append(u.squeeze(0).squeeze(-1).cpu().numpy())               # t=1
    # Keep rolling:
    for _ in range(Nt-1):
        # append new frame to window: drop oldest, add u
        x_win = torch.cat([x_win[..., 1:], u], dim=-1)
        u = model(x_win)
        preds.append(u.squeeze(0).squeeze(-1).cpu().numpy())
    return np.stack(preds, axis=0)  # (Nt+1, S,S,S)

# -----------------------------
# 6) Pretty isosurface helper
# -----------------------------
def add_isosurface(ax, vol, iso=0.0):
    """Compute an isosurface and add it to a 3D axis with clean styling."""
    verts, faces, _, _ = measure.marching_cubes(vol, level=iso)
    mesh = Poly3DCollection(verts[faces])
    # keep it minimal but pleasant; let matplotlib choose defaults
    mesh.set_edgecolor('none')
    ax.add_collection3d(mesh)
    ax.set_box_aspect([1,1,1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # tight bounds
    s = vol.shape[0]
    ax.set_xlim(0, s); ax.set_ylim(0, s); ax.set_zlim(0, s)
    ax.view_init(elev=22, azim=45)  # gentle cinematic angle

# -----------------------------
# 7) Main: eval + plot grid
# -----------------------------
def evaluate_and_plot(
    which_ic="sphere", frames=(20, 60, 100), save_path=None, figsize_per_col=4.0
):
    assert which_ic in ("sphere", "star")
    if which_ic == "sphere":
        u0 = ic_sphere()
    else:
        u0 = ic_star()

    # Seed window
    x0 = seed_window_from_ic(u0, T_in=CFG.T_IN_CHANNELS, device=CFG.DEVICE)

    # Load models and roll
    methods = list(CKPTS.keys())  # order: FNO4d, MHNO, PENCO, PurePhysics
    preds = {}
    for m in methods:
        model_type, path = CKPTS[m]
        model = build_model(model_type)
        model = load_checkpoint(model, path)
        preds[m] = rollout_from_window(model, x0, Nt=CFG.TOTAL_TIME_STEPS)

    # Figure size rule: 4 * len(frames)
    n_rows = len(methods)
    n_cols = len(frames)
    fig_w = figsize_per_col * n_cols
    fig_h = 3.2 * n_rows  # tall enough for labels and aspect
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.02, hspace=0.02)

    # Column labels on top
    for j, t in enumerate(frames):
        fig.text(
            x=(j + 0.5) / n_cols,
            y=0.98,
            s=f"{t} / Δt",
            ha="center",
            va="top",
            fontsize=12,
        )

    # Row labels on left margin
    for i, m in enumerate(methods):
        fig.text(
            x=0.01,
            y=(n_rows - i - 0.5) / n_rows,
            s=m,
            ha="left",
            va="center",
            rotation=90,
            fontsize=12,
        )

    # Populate cells
    for i, m in enumerate(methods):
        P = preds[m]  # (Nt+1, S,S,S)
        for j, t in enumerate(frames):
            ax = fig.add_subplot(gs[i, j], projection="3d")
            add_isosurface(ax, P[t], iso=0.0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # Choose one IC at a time (or call twice)
    # 1) Smooth sphere
    evaluate_and_plot(which_ic="sphere", frames=(20, 60, 100),
                      save_path=None, figsize_per_col=4.0)

    # 2) Spiky star (uncomment to also visualize)
    # evaluate_and_plot(which_ic="star", frames=(20, 60, 100),
    #                   save_path=None, figsize_per_col=4.0)
