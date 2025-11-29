"""
Evaluate FNO4d / MHNO / PENCO-FNO / PENCO-MHNO / PurePhysics
on test .mat datasets for AC3D, CH3D, SH3D, PFC3D, MBE3D.

For each PDE:
    - load ICs from the provided test .mat file
    - for each sample (you can restrict to the first one if you like)
    - bootstrap teacher window by semi-implicit PDE steps
    - run each trained model autoregressively
    - save everything into a MATLAB .mat file:

        <PROBLEM>_test_eval_for_boxplot.mat

    Contents:
        meta: struct (dt, Nt, grid sizes, L-domain, epsilon, time_frames)
        U0_all:  [Nx, Ny, Nz, Nsamples]
        predictions: struct with fields
            FNO4d, MHNO, PENCO_MHNO, PENCO_FNO, PurePhysics
            each of size [Nt+1, Nx, Ny, Nz, Nsamples]
        method_order: cellstr of original labels
        method_fieldnames: cellstr of MATLAB-safe struct field names
"""

import os
import numpy as np
import torch
import scipy.io as sio
import torch.fft as tfft
import h5py  # <-- add this

# --- repo imports (must be importable from this script) ---
import config as CFG
from networks import FNO4d, TNO3d
from functions import (
    semi_implicit_step,        # AC3D
    semi_implicit_step_pfc,    # PFC3D
    semi_implicit_step_mbe     # MBE3D
)

# ==========================================================
# 0) COMMON SETTINGS / PATHS
# ==========================================================

DEVICE = CFG.DEVICE
T_IN = CFG.T_IN_CHANNELS

# Map each problem to its checkpoint dict (you already have these)
from OOD_phase_field_equations import (
    CKPTS_AC3D, CKPTS_CH3D, CKPTS_SH3D,
    CKPTS_PFC3D, CKPTS_MBE3D
)

CKPTS_BY_PROBLEM = {
    'AC3D': CKPTS_AC3D,
    'CH3D': CKPTS_CH3D,
    'SH3D': CKPTS_SH3D,
    'PFC3D': CKPTS_PFC3D,
    'MBE3D': CKPTS_MBE3D,
}

METHODS = ["FNO4d", "MHNO", "PENCO-MHNO", "PENCO-FNO", "PurePhysics"]

# Test data paths you gave
TEST_DATA_PATHS = {
    'AC3D': "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/AC3D_32_10_grf3d.mat",
    'CH3D': "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/CH3D_10_Nt_101_Nx_32.mat",
    'SH3D': "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/SH3D_grf3d_ff_10_Nt_101_Nx_32.mat",
    'PFC3D': "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/PFC3D_Augmented_10_Nt_101_Nx_32.mat",
    'MBE3D': "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/MBE3D_Augmented_10_Nt_101_Nx_32.mat",
}

# Time frames of interest for boxplot
TIME_FRAMES = np.array([0, 10, 20, 30, 80, 90, 100], dtype=np.int32)

# ==========================================================
# 1) PDE-specific semi-implicit steps (CH3D, SH3D)
# ==========================================================
def load_mat_any(path):
    """
    Load a MATLAB .mat file of any version.
    - For v5/v6/v7: uses scipy.io.loadmat
    - For v7.3 (HDF5): falls back to h5py and returns a dict-like
      mapping from dataset names to numpy arrays.
    """
    try:
        return sio.loadmat(path)
    except NotImplementedError:
        # v7.3 HDF5 path
        md = {}

        with h5py.File(path, "r") as f:
            # collect ALL datasets recursively into a flat dict
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # obj[()] reads the whole dataset as a numpy array
                    md[name] = np.array(obj[()])
            f.visititems(visitor)

        return md

def semi_implicit_step_ch3d(u, dt, dx, epsilon):
    """
    CH3D semi-implicit (matches MATLAB):
        u^{n+1} = ifftn( (Û - dt*K2*F[u^3 - 3u]) / (1 + dt*(2*K2 + eps^2*K2^2)) )
    u: (B,S,S,S,1), real
    """
    u = u[..., 0]                       # (B,Sx,Sy,Sz)
    B, Sx, Sy, Sz = u.shape

    # freq vectors
    kx = 2 * np.pi * np.fft.fftfreq(Sx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Sy, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Sz, d=dx)

    KX2 = torch.from_numpy(kx.astype(np.float32) ** 2).to(u.device)
    KY2 = torch.from_numpy(ky.astype(np.float32) ** 2).to(u.device)
    KZ2 = torch.from_numpy(kz.astype(np.float32) ** 2).to(u.device)
    K2x, K2y, K2z = torch.meshgrid(KX2, KY2, KZ2, indexing="ij")
    K2 = K2x + K2y + K2z

    U = tfft.fftn(u, dim=(-3, -2, -1))
    nonlin = u ** 3 - 3.0 * u
    NL = tfft.fftn(nonlin, dim=(-3, -2, -1))

    numer = U - dt * K2 * NL
    denom = 1.0 + dt * (2.0 * K2 + (epsilon ** 2) * (K2 ** 2))
    V = numer / denom

    u_next = tfft.ifftn(V, dim=(-3, -2, -1)).real
    return u_next.unsqueeze(-1)


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

    KX2 = torch.from_numpy((kx.astype(np.float32)) ** 2).to(u.device)
    KY2 = torch.from_numpy((ky.astype(np.float32)) ** 2).to(u.device)
    KZ2 = torch.from_numpy((kz.astype(np.float32)) ** 2).to(u.device)
    K2x, K2y, K2z = torch.meshgrid(KX2, KY2, KZ2, indexing="ij")
    K2 = K2x + K2y + K2z

    U = tfft.fftn(u, dim=(-3, -2, -1))
    NL = tfft.fftn(u ** 3, dim=(-3, -2, -1))

    numer = (U / dt) - NL + 2.0 * K2 * U
    denom = (1.0 / dt) + (1.0 - float(epsilon)) + (K2 ** 2)

    V = numer / denom
    u_next = tfft.ifftn(V, dim=(-3, -2, -1)).real
    return u_next.unsqueeze(-1)

# ==========================================================
# 2) Helpers: load models, bootstrap window, rollout
# ==========================================================

def load_model(method: str, ckpt_dict, device: torch.device):
    """
    Load one model for a given method (FNO4d or TNO3d), inferring
    width_q (and width_h for TNO3d) from the checkpoint so we can
    evaluate multiple problems even if CFG.PROBLEM is different.
    """
    model_type, path = ckpt_dict[method]
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"]

    if model_type == "FNO4d":
        # --- infer width_q from checkpoint ---
        wq = sd.get("q.layers.0.weight", None)
        if wq is not None:
            width_q = int(wq.shape[0])   # e.g. 10 for AC3D, 12 for MBE3D
        else:
            width_q = getattr(CFG, "WIDTH_Q", CFG.WIDTH)

        model = FNO4d(
            modes1=CFG.MODES,
            modes2=CFG.MODES,
            modes3=CFG.MODES,
            modes4_internal=None,
            width=CFG.WIDTH,          # this already matches all your runs
            width_q=width_q,
            T_in_channels=CFG.T_IN_CHANNELS,
            n_layers=CFG.N_LAYERS
        ).to(device)

    elif model_type == "TNO3d":
        # --- infer widths from checkpoint (you already had this) ---
        wq = sd.get("q.layers.0.weight", None)
        wh = sd.get("h.layers.0.weight", None)
        width_q = int(wq.shape[0]) if wq is not None else CFG.WIDTH_Q
        width_h = int(wh.shape[0]) if wh is not None else CFG.WIDTH_H

        model = TNO3d(
            modes1=CFG.MODES,
            modes2=CFG.MODES,
            modes3=CFG.MODES,
            width=CFG.WIDTH,
            width_q=width_q,
            width_h=width_h,
            T_in=CFG.T_IN_CHANNELS,
            T_out=1,
            n_layers=CFG.N_LAYERS
        ).to(device)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(sd)
    model.eval()
    return model.to(device)



@torch.no_grad()
def bootstrap_from_u0(u0_np, problem, spec, device):
    """
    Build teacher window of length T_IN using the semi-implicit numerical
    scheme appropriate for the problem.
    Returns:
      teacher_states: list of tensors [ (1,Nx,Ny,Nz,1) ... ]
      x0:             input tensor (1,Nx,Ny,Nz,T_IN)
    """
    dt = float(spec['DT'])
    dx = spec['L_DOMAIN'] / spec['GRID_RESOLUTION']
    eps = float(spec['EPSILON_PARAM'])
    eps2 = eps ** 2

    u = torch.from_numpy(u0_np.astype(np.float32)).to(device=device)
    u = u.unsqueeze(0).unsqueeze(-1)  # (1,Nx,Ny,Nz,1)
    states = [u]

    for _ in range(1, T_IN):
        if problem == 'AC3D':
            u = semi_implicit_step(u, dt, dx, eps2)
        elif problem == 'CH3D':
            u = semi_implicit_step_ch3d(u, dt, dx, eps)
        elif problem == 'SH3D':
            u = semi_implicit_step_sh3d(u, dt, dx, eps)
        elif problem == 'PFC3D':
            u = semi_implicit_step_pfc(u, dt, dx, eps)
        elif problem == 'MBE3D':
            u = semi_implicit_step_mbe(u, dt, dx, eps)
        else:
            raise ValueError(f"Unknown problem '{problem}'")
        states.append(u)

    x0 = torch.cat(states, dim=-1)  # (1,Nx,Ny,Nz,T_IN)
    return states, x0


@torch.no_grad()
def rollout_aligned(model, x0, teacher_states, Nt):
    """
    Roll out model autoregressively to Nt steps.
    Returns vols: (Nt+1, Nx,Ny,Nz) numpy float32
    """
    T_in = x0.shape[-1]
    seq = [st.squeeze(0).squeeze(-1).cpu().numpy() for st in teacher_states]

    x = x0.clone()
    steps_to_predict = Nt + 1 - T_in

    for _ in range(steps_to_predict):
        y_pred = model(x)  # (1,Nx,Ny,Nz,1)
        seq.append(y_pred.squeeze(0).squeeze(-1).cpu().numpy())
        x = torch.cat([x[..., 1:], y_pred], dim=-1)

    return np.stack(seq, axis=0).astype(np.float32)


# ==========================================================
# 3) Load ICs from test .mat data
# ==========================================================

def find_state_array(md, Nx, Nt_plus1):
    """
    Heuristic: find an ndarray in the .mat dict whose shape contains
    one dimension == Nt+1 and *three* dimensions == Nx (32).
    Adjust this if your variable name is known (e.g. replace with md['u']).
    """
    for key, val in md.items():
        if not isinstance(val, np.ndarray):
            continue
        shp = val.shape
        if (Nt_plus1 in shp) and (list(shp).count(Nx) >= 3) and val.ndim >= 4:
            print(f"  Using variable '{key}' with shape {shp}")
            return val
    raise RuntimeError("Could not find a suitable state array in .mat file; "
                       "please adjust 'find_state_array' to use the right key.")


def make_u0_extractor(mat_path, spec):
    """
    Returns:
      get_u0(sample_idx) -> u0 (Nx,Ny,Nz) float32
      n_samples
    Assumes the .mat file contains a 4D or 5D array with dims including:
         - Nt+1 (time)
         - 32,32,32 (space)
         - (#samples)  (optional extra dim)
    """
    #md = sio.loadmat(mat_path)
    md = load_mat_any(mat_path)
    Nx = spec['GRID_RESOLUTION']
    Nt_plus1 = spec['TOTAL_TIME_STEPS'] + 1

    arr = find_state_array(md, Nx, Nt_plus1)
    shp = list(arr.shape)

    # identify axes
    time_axis = [i for i, s in enumerate(shp) if s == Nt_plus1][0]
    space_axes = [i for i, s in enumerate(shp) if s == Nx]
    other_axes = [i for i in range(len(shp)) if i not in space_axes + [time_axis]]

    if len(space_axes) < 3:
        raise RuntimeError("Less than 3 spatial Nx dimensions found. "
                           "Adjust 'make_u0_extractor' to your data format.")

    # assume only one remaining axis = sample axis
    if len(other_axes) == 0:
        sample_axis = None
        n_samples = 1
    else:
        sample_axis = other_axes[0]
        n_samples = shp[sample_axis]

    print(f"  time_axis={time_axis}, space_axes={space_axes}, "
          f"sample_axis={sample_axis}, n_samples={n_samples}")

    def get_u0(sample_idx):
        idx = [slice(None)] * arr.ndim
        idx[time_axis] = 0  # t=0
        if sample_axis is not None:
            idx[sample_axis] = sample_idx
        u0_raw = arr[tuple(idx)]

        # ensure we end with (Nx,Ny,Nz)
        u0 = np.array(u0_raw, dtype=np.float32)
        if u0.shape != (Nx, Nx, Nx):
            # pick the three Nx-dims and reorder
            axes = [i for i, s in enumerate(u0.shape) if s == Nx]
            if len(axes) != 3:
                raise RuntimeError(f"Unexpected u0 shape {u0.shape} for sample {sample_idx}")
            u0 = np.transpose(u0, axes)
        return u0

    return get_u0, n_samples


# ==========================================================
# 4) MAIN EVALUATION
# ==========================================================

def method_fieldname(method_label: str) -> str:
    """Convert method label to a MATLAB-safe struct field name."""
    return method_label.replace('-', '_')


def evaluate_problem(problem: str, n_eval_samples: int = 1):
    """
    Evaluate all METHODS on 'problem' using the corresponding test data.
    Only the first n_eval_samples in the .mat file are used.
    Produces <problem>_test_eval_for_boxplot.mat
    """
    print(f"\n=== Evaluating problem: {problem} ===")
    spec = CFG.PROBLEM_SPECS[problem]
    Nx = spec['GRID_RESOLUTION']
    Nt = spec['TOTAL_TIME_STEPS']
    dt = float(spec['DT'])
    eps = float(spec['EPSILON_PARAM'])

    mat_path = TEST_DATA_PATHS[problem]
    assert os.path.exists(mat_path), f"Missing test data file: {mat_path}"

    # Build extractor for ICs
    get_u0, n_samples_total = make_u0_extractor(mat_path, spec)
    n_samples = min(n_eval_samples, n_samples_total)
    print(f"  Will evaluate {n_samples} sample(s) out of {n_samples_total}")

    # Prepare storage: ICs and prediction volumes
    U0_all = np.zeros((Nx, Nx, Nx, n_samples), dtype=np.float32)

    predictions = {}
    for m in METHODS:
        field = method_fieldname(m)
        predictions[field] = np.zeros((Nt + 1, Nx, Nx, Nx, n_samples), dtype=np.float32)

    # Pre-load all models once
    ckpts = CKPTS_BY_PROBLEM[problem]
    models = {}
    for m in METHODS:
        print(f"  Loading model: {problem} / {m}")
        models[m] = load_model(m, ckpts, DEVICE)

    # Evaluate each sample
    for s in range(n_samples):
        print(f"  Sample {s+1} / {n_samples}")
        u0_np = get_u0(s)  # (Nx,Nx,Nx)
        U0_all[:, :, :, s] = u0_np

        teacher_states, x0 = bootstrap_from_u0(u0_np, problem, spec, DEVICE)

        for m in METHODS:
            vols = rollout_aligned(models[m], x0, teacher_states, Nt=Nt)  # (Nt+1,Nx,Ny,Nz)
            field = method_fieldname(m)
            predictions[field][:, :, :, :, s] = vols

    # Clear models to free GPU
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build MATLAB-friendly dict
    meta = {
        'case_study': problem,
        'grid_sizes': np.array([Nx, Nx, Nx], dtype=np.int32),
        'domain_lengths': np.array([spec['L_DOMAIN']] * 3, dtype=np.float32),
        'dx': np.float32(spec['L_DOMAIN'] / Nx),
        'dt': np.float32(dt),
        'Nt': np.int32(Nt),
        'epsilon': np.float32(eps),
        'time_frames': TIME_FRAMES.astype(np.int32),
    }

    out = {
        'meta': meta,
        'U0_all': U0_all,
        'predictions': predictions,
        'method_order': np.array(METHODS, dtype=object),
        'method_fieldnames': np.array(
            [method_fieldname(m) for m in METHODS], dtype=object
        ),
    }

    out_name = f"{problem}_test_eval_for_boxplot.mat"
    sio.savemat(out_name, out, do_compression=True)
    print(f"  Saved MATLAB data to: {out_name}")


if __name__ == "__main__":
    # Change n_eval_samples if you want to use more than one IC per PDE
    N_EVAL_SAMPLES = 10  # you can set to 1 if you truly want one IC

    for prob in ['AC3D', 'CH3D', 'SH3D', 'PFC3D', 'MBE3D']:
        evaluate_problem(prob, n_eval_samples=N_EVAL_SAMPLES)
