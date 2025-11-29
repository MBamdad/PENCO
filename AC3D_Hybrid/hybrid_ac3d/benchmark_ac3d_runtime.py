# benchmark_ac3d_runtime.py
#
# Fair runtime comparison: AC3D numerical solver vs trained models
#
# - Same IC, Nt, grid, device for ALL
# - Pure float32 (no AMP) for both solver and models
# - Optional torch.compile() for models (off by default)
#
# Usage:
#   python benchmark_ac3d_runtime.py
#

import time
from pathlib import Path

import numpy as np
import torch

import config as CFG
from networks import FNO4d, TNO3d

# ================== SETTINGS ==================
USE_TORCH_COMPILE = False  # you can try True later; keep False for robustness
N_REPS = 5                # timing repetitions
# ==============================================

# ------------------------------------------------------------
# Checkpoints for AC3D (your paths)
# ------------------------------------------------------------
CKPTS_AC3D = {
    # method_label: (model_type, path)
    "FNO4d": (
        "FNO4d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/FNO4d_FNO4d_N100_pw0.00_E50.pt",
    ),
    "MHNO": (
        "TNO3d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/TNO3d_MHNO_N200_pw0.00_E50.pt",
    ),
    "PENCO-MHNO": (
        "TNO3d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/TNO3d_PENCO_N200_pw0.50_E50.pt",
    ),
    "PENCO-FNO": (
        "FNO4d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/FNO4d_PENCO_N200_pw0.50_E50.pt",
    ),
    "PurePhysics": (
        "TNO3d",
        "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/AC3d_models/TNO3d_PurePhysics_N200_pw1.00_E50.pt",
    ),
}


# ============================================================
# 1) Load model
# ============================================================
def load_model_from_ckpt(method_label: str, device: torch.device):
    model_type, ckpt_path = CKPTS_AC3D[method_label]
    ckpt_path = Path(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)

    if model_type == "FNO4d":
        model = FNO4d(
            modes1=CFG.MODES,
            modes2=CFG.MODES,
            modes3=CFG.MODES,
            modes4_internal=None,
            width=CFG.WIDTH,
            width_q=CFG.WIDTH_Q,
            T_in_channels=CFG.T_IN_CHANNELS,
            n_layers=CFG.N_LAYERS,
        )
    else:
        model = TNO3d(
            modes1=CFG.MODES,
            modes2=CFG.MODES,
            modes3=CFG.MODES,
            width=CFG.WIDTH,
            width_q=CFG.WIDTH_Q,
            width_h=CFG.WIDTH_H,
            T_in=CFG.T_IN_CHANNELS,
            T_out=1,
            n_layers=CFG.N_LAYERS,
        )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
        except Exception:
            print(f"[WARN] torch.compile failed for {method_label}, continuing without it.")
    return model


# ============================================================
# 2) GRF-like IC
# ============================================================
def make_random_ic(nx, ny, nz, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(nx, ny, nz)).astype(np.float32)

    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kz = np.fft.fftfreq(nz)
    kxx, kyy, kzz = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kxx**2 + kyy**2 + kzz**2
    k2 = np.where(k2 == 0, 1.0, k2)
    sigma = 0.10
    filt = np.exp(-k2 / (sigma**2))

    noise_hat = np.fft.fftn(noise)
    smooth = np.fft.ifftn(noise_hat * filt).real.astype(np.float32)

    u0 = np.ones_like(smooth, dtype=np.float32)
    u0[smooth < 0] = -1.0
    return u0


# ============================================================
# 3) AC3D spectral solver (float32)
# ============================================================
def build_k2_grid(nx, ny, nz, dx, device: torch.device):
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
        k2 = build_k2_grid(nx, ny, nz, dx, device)

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


def time_ac3d_numerical_solver(
    u0_np, dt, dx, eps2, Nt, device: torch.device, k2=None, n_reps=5
):
    nx, ny, nz = u0_np.shape
    Cahn = eps2

    if k2 is None:
        k2 = build_k2_grid(nx, ny, nz, dx, device)

    u0 = torch.from_numpy(u0_np.astype(np.float32)).to(device)

    # warm-up
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


# ============================================================
# 4) Torch-only autoregressive rollout for models (float32)
# ============================================================
def rollout_autoregressive_torch(
    model, traj_np, T_in, Nt, device: torch.device
):
    assert isinstance(traj_np, np.ndarray)
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
            y_pred = y_pred[0]        # (Sx,Sy,Sz)

            pred[t] = y_pred

    return pred.detach().cpu().numpy().astype(np.float32)


def time_model_rollout(
    model, traj_np, T_in, Nt, device: torch.device, n_reps=5
):
    # warm-up
    _ = rollout_autoregressive_torch(model, traj_np, T_in, Nt, device)

    times = []
    for _ in range(n_reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = rollout_autoregressive_torch(model, traj_np, T_in, Nt, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        times.append(t2 - t1)

    return float(np.mean(times)), float(np.std(times))


# ============================================================
# 5) Main
# ============================================================
def main():
    device = CFG.DEVICE if isinstance(CFG.DEVICE, torch.device) else torch.device(CFG.DEVICE)
    print(f"Device: {device}")
    print(f"Models: {', '.join(CKPTS_AC3D.keys())}")

    Nt = CFG.TOTAL_TIME_STEPS
    T_in = CFG.T_IN_CHANNELS
    dx = CFG.DX
    dt = CFG.DT
    eps2 = CFG.EPS2
    nx = ny = nz = CFG.GRID_RESOLUTION

    print(f"Grid: {nx}^3, Nt={Nt}, dt={dt}, dx={dx}, eps2={eps2}")

    # Shared IC + k^2 grid
    u0 = make_random_ic(nx, ny, nz, seed=0)
    k2 = build_k2_grid(nx, ny, nz, dx, device)

    # Exact AC3D trajectory
    traj_exact = ac3d_solve_full_trajectory(
        u0, dt, dx, eps2, Nt, device, k2=k2
    )

    # Time solver
    solver_mean, solver_std = time_ac3d_numerical_solver(
        u0, dt, dx, eps2, Nt, device, k2=k2, n_reps=N_REPS
    )

    print("\n===== AC3D Numerical Solver (reference) =====")
    print(f"AC3D numerical solver: {solver_mean:.4f} s ± {solver_std:.4f} s\n")

    results = {}

    for label in CKPTS_AC3D.keys():
        print(f"--- Benchmarking model: {label} ---")
        model = load_model_from_ckpt(label, device)

        mean_model, std_model = time_model_rollout(
            model,
            traj_exact,
            T_in,
            Nt,
            device,
            n_reps=N_REPS,
        )

        speedup = solver_mean / mean_model if mean_model > 0 else float("nan")
        results[label] = (mean_model, std_model, speedup)

        print(f"Model rollout ({label}): {mean_model:.4f} s ± {std_model:.4f} s")
        print(f"Speedup (solver / {label}): {speedup:.3f}×\n")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("===== Summary: model vs AC3D solver (same IC, same Nt, same grid, same device) =====")
    print(f"AC3D solver: {solver_mean:.4f} s ± {solver_std:.4f} s")
    for label, (m_mean, m_std, sp) in results.items():
        print(
            f"{label:12s}  model: {m_mean:.4f} s ± {m_std:.4f} s   "
            f"speedup (solver/model): {sp:.3f}×"
        )


if __name__ == "__main__":
    main()
