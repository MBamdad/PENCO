import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.io import savemat
from scipy.ndimage import zoom as nd_zoom  # for smooth upsampling

# --- repo imports (your files) ---
import config as CFG
from networks import FNO4d, TNO3d
from functions import semi_implicit_step

# -----------------------------
# 1) Paths to your checkpoints
# -----------------------------
CKPTS = {
    "FNO4d":      "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/FNO4d_FNO4d_N200_pw0.00_E50.pt",
    "MHNO":       "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/TNO3d_MHNO_N200_pw0.00_E50.pt",
    # RENAMED: previous 'PENCO' -> 'PENCO-MHNO' (TNO3d backbone)
    "PENCO-MHNO": "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/TNO3d_PENCO_N200_pw0.25_E50.pt",
    # ADDED: FNO4d + PENCO
    "PENCO-FNO":  "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/FNO4d_PENCO_N200_pw0.25_E50.pt",
    "PurePhysics":"/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models/AC3D/TNO3d_PurePhysics_N200_pw1.00_E50.pt",
}

# -----------------------------------
# 2) Sphere initial condition (AC3D)
# -----------------------------------
def create_initial_condition_sphere_ac3d():
    Nx = Ny = Nz = 32
    Nt = 100
    Lx = Ly = Lz = 2.0
    epsilon = 0.1
    dt = float(CFG.DT)             # single source of truth for dt
    selected_frames = [0, 20, 40, 60, 80, 100]

    radius = 0.5
    x_grid = np.linspace(-Lx/2, Lx/2, Nx)
    y_grid = np.linspace(-Ly/2, Ly/2, Ny)
    z_grid = np.linspace(-Lz/2, Lz/2, Nz)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    interface_width = np.sqrt(2.0) * epsilon
    u0 = np.tanh((radius - np.sqrt(xx**2 + yy**2 + zz**2)) / interface_width).astype(np.float32)

    return u0, (Lx, Ly, Lz), (Nx, Ny, Nz), Nt, dt, selected_frames

# ----------------------------------------------------
# 3) Model loading (match training hyperparameters)
# ----------------------------------------------------
def load_model(kind: str, device: torch.device):
    """
    Select architecture by method:
      - FNO4d backbone: 'FNO4d', 'PENCO-FNO'
      - TNO3d backbone: 'MHNO', 'PENCO-MHNO', 'PurePhysics'
    """
    if kind in ("FNO4d", "PENCO-FNO"):
        model = FNO4d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES, modes4_internal=None,
            width=CFG.WIDTH, width_q=CFG.WIDTH_Q, T_in_channels=CFG.T_IN_CHANNELS,
            n_layers=CFG.N_LAYERS
        ).to(device)
    else:
        model = TNO3d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES,
            width=CFG.WIDTH, width_q=CFG.WIDTH_Q, width_h=CFG.WIDTH_H,
            T_in=CFG.T_IN_CHANNELS, T_out=1, n_layers=CFG.N_LAYERS
        ).to(device)

    ckpt = torch.load(CKPTS[kind], map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# -----------------------------------------------------------
# 4) Bootstrap T_in window with semi-implicit teacher
# -----------------------------------------------------------
@torch.no_grad()
def bootstrap_states_from_u0(u0_np: np.ndarray, T_in: int, device: torch.device):
    """
    Returns:
      states: list of T_in tensors [u^0, u^1, ..., u^{T_in-1}] each (1,S,S,S,1)
      x0:     window tensor (1,S,S,S,T_in) built from states
    """
    u = torch.from_numpy(u0_np).to(device=device).unsqueeze(0).unsqueeze(-1)  # (1,S,S,S,1)
    states = [u]  # u^0
    for _ in range(1, T_in):
        u = semi_implicit_step(u, CFG.DT, CFG.DX, CFG.EPS2)
        states.append(u)  # u^1 ... u^{T_in-1}
    x0 = torch.cat(states, dim=-1)  # (1,S,S,S,T_in)
    return states, x0

# -----------------------------------------------------------
# 5) Rollout autoregressively from the bootstrapped window
# -----------------------------------------------------------
@torch.no_grad()
def rollout_aligned(model, x0: torch.Tensor, teacher_states: list, Nt: int):
    """
    Produces an aligned sequence U[0..Nt], where:
      U[0..T_in-1]  = teacher_states (u^0 ... u^{T_in-1})
      U[T_in..Nt]   = model predictions (autoregressive) continuing from window x0
    Shapes:
      - teacher_states: list of T_in tensors, each (1,S,S,S,1)
      - x0: (1,S,S,S,T_in)
      - returns: np.array of shape (Nt+1, S, S, S)
    """
    device = x0.device
    T_in = x0.shape[-1]

    # Start the stored sequence with the teacher frames (aligned with u0)
    seq = [st.squeeze(0).squeeze(-1).detach().cpu().numpy() for st in teacher_states]  # length T_in

    # Autoregressive predictions for the remaining steps
    x = x0.clone()
    steps_to_predict = Nt + 1 - T_in
    for _ in range(steps_to_predict):
        y_pred = model(x)                       # (1,S,S,S,1)
        seq.append(y_pred.squeeze(0).squeeze(-1).cpu().numpy())
        x = torch.cat([x[..., 1:], y_pred], dim=-1)

    return np.stack(seq, axis=0).astype(np.float32)  # (Nt+1,S,S,S)

# -----------------------------------------------------------
# 6) High-quality 3D plotting (paper-ready)
# -----------------------------------------------------------
def plot_isosurface_grid(volumes_by_method, frames, out_png="ac3d_compare.png", out_pdf="ac3d_compare.pdf",
                         upsample=2, pad_vox=2):
    """
    volumes_by_method: dict name -> array (Nt+1,S,S,S)
    frames: list of time indices to visualize
    - upsample: trilinear upsampling factor for smoother surfaces
    - pad_vox: padding (in upsampled voxels) around the tight bbox
    """
    # Show all five methods, in this order:
    method_order = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]
    n_rows = len(method_order)
    n_cols = len(frames)

    # big canvas; lots of dpi
    fig_w = 4.8 * n_cols
    fig_h = 4.2 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.05, wspace=0.05, hspace=0.05)

    # vivid, bright, no edges — give PENCO-FNO a warm orange
    facecolor_map = {
        "FNO4d":       (1.00, 0.10, 0.10, 1.0),  # bright red
        "MHNO":        (0.95, 0.20, 0.20, 1.0),  # warm red
        "PENCO-MHNO":  (0.10, 0.90, 0.10, 1.0),  # bright green
        "PENCO-FNO":   (0.95, 0.50, 0.15, 1.0),  # warm orange
        "PurePhysics": (0.05, 0.95, 0.05, 1.0),  # bright green
    }

    # helper for per-panel zoom: compute bbox from vertices
    def _tighten_axes(ax, verts, pad=pad_vox):
        if verts.size == 0:
            return
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        (z0, y0, x0) = mins
        (z1, y1, x1) = maxs
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_zlim(z0 - pad, z1 + pad)

    plot_idx = 1
    for r, method in enumerate(method_order):
        vols = volumes_by_method[method]
        for c, t in enumerate(frames):
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            plot_idx += 1
            ax.set_facecolor('white')
            ax.grid(False)

            vol = vols[t]                     # (X,Y,Z)
            vol_zyx = np.transpose(vol, (2,1,0))  # (Z,Y,X) for marching_cubes

            # upsample smoothly in (Z,Y,X) for a higher-res surface
            if upsample and upsample > 1:
                # order=1 trilinear; preserve sign nicely for level=0
                vol_zyx = nd_zoom(vol_zyx, zoom=upsample, order=1)

            try:
                verts, faces, _, _ = marching_cubes(vol_zyx, level=0.0, step_size=1, allow_degenerate=False)
                rgba = facecolor_map[method]
                mesh = Poly3DCollection(verts[faces], alpha=rgba[3], linewidth=0.0, antialiased=True)
                mesh.set_facecolor(rgba)
                mesh.set_edgecolor('none')  # no mesh lines
                ax.add_collection3d(mesh)
                _tighten_axes(ax, verts, pad=pad_vox)
            except Exception as e:
                ax.text2D(0.05, 0.90, f"MC fail @ t={t}\n{e}", transform=ax.transAxes, color='k', fontsize=10)

            # only put the delta-t column headers on the top row
            if r == 0:
                ax.set_title(rf"${t}\,\Delta t$", fontsize=20, fontweight='bold', pad=8)

            # clean axes for a minimal look
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=22, azim=-60)

        # left-side vertical method labels (once per row)
        y_center = 1.0 - (r + 0.5) / n_rows
        fig.text(0.03, y_center, method, va='center', ha='center',
                 rotation=90, fontsize=22, fontweight='bold')

    plt.savefig(out_png, dpi=400, facecolor='white')
    plt.savefig(out_pdf, dpi=400, facecolor='white')
    plt.close(fig)
    print(f"Saved comparison figures to: {out_png} and {out_pdf}")

# -----------------------------------------------------------
# 7) Main: load, rollout, plot, save .mat
# -----------------------------------------------------------
def main():
    device = CFG.DEVICE

    # IC
    u0_np, domain_lengths, grid_sizes, Nt, dt, selected_frames = create_initial_condition_sphere_ac3d()
    Sx, Sy, Sz = grid_sizes
    assert (Sx, Sy, Sz) == (CFG.GRID_RESOLUTION, CFG.GRID_RESOLUTION, CFG.GRID_RESOLUTION), \
        f"Grid mismatch: IC {grid_sizes} vs CFG.GRID_RESOLUTION={CFG.GRID_RESOLUTION}"
    assert abs(dt - CFG.DT) < 1e-12, "Python saver dt != CFG.DT (time-step mismatch)"
    # Also ensure DX and EPS2 match your MATLAB DNS choices.

    # Build aligned teacher window
    teacher_states, x0 = bootstrap_states_from_u0(u0_np, CFG.T_IN_CHANNELS, device=device)

    # load models & rollout (aligned to u0)
    # NOTE: now five methods including both PENCO variants
    methods = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]

    volumes_by_method = {}
    for method in methods:
        print(f"Loading & rolling out: {method}")
        model = load_model(method, device=device)
        vols = rollout_aligned(model, x0, teacher_states, Nt=Nt)  # (Nt+1,S,S,S), U[0]=u0
        volumes_by_method[method] = vols
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # optional visualization
    plot_isosurface_grid(
        volumes_by_method, selected_frames,
        out_png="ac3d_sphere_methods_compare.png",
        out_pdf="ac3d_sphere_methods_compare.pdf",
        upsample=3,
        pad_vox=3
    )

    # save .mat — sequences start at u0 (index 0) and match MATLAB DNS frames
    out = {
        'meta': {
            'case': 'AC3D_sphere',
            'grid_sizes': np.array(grid_sizes, dtype=np.int32),
            'domain_lengths': np.array(domain_lengths, dtype=np.float32),
            'dx': np.float32(CFG.DX),
            'dt': np.float32(dt),
            'Nt': np.int32(Nt),
            'T_in_channels': np.int32(CFG.T_IN_CHANNELS),
            'selected_frames': np.array(selected_frames, dtype=np.int32),
        },
        'FNO4d':        volumes_by_method['FNO4d'],         # (Nt+1,S,S,S)
        'MHNO':         volumes_by_method['MHNO'],
        'PENCO_MHNO':   volumes_by_method['PENCO-MHNO'],   # use underscore to be MATLAB-friendly
        'PENCO_FNO':    volumes_by_method['PENCO-FNO'],
        'PurePhysics':  volumes_by_method['PurePhysics'],
        'teacher_window': np.stack([s.squeeze(0).squeeze(-1).cpu().numpy() for s in teacher_states], axis=0),
        'U0': u0_np,
    }
    mat_name = "AC3D_sphere_eval_compare.mat"
    savemat(mat_name, out, do_compression=True)
    print(f"Saved MATLAB results to: {mat_name}")

if __name__ == "__main__":
    main()
