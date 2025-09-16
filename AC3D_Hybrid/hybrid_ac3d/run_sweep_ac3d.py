# run_sweep_ac3d.py
# Runs all combos, logs metrics, saves models + a single .mat file for MATLAB plotting.
# Does NOT modify your repo's code; it reuses/imports it.

import os, sys, json, time, random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.io import savemat

import config as CFG
from networks import FNO4d, TNO3d
from utilities import (
    build_loaders, relative_l2, rollout_autoregressive
)
from functions import (
    physics_residual_midpoint, physics_residual_normalized, scheme_residual_fourier,
    loss_mm_projection, semi_implicit_step, energy_penalty, mu_ac
)

# --------------------
# Grid you requested
# --------------------
PDE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODELS = ['FNO4d', 'TNO3d']  # TNO3d will be labeled "MHNO" in saved results
#N_TRAINS = [50, 100, 150, 200]
N_TRAINS = [50, 100, 200]

# Where to save
MODELS_DIR = Path("/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Models")
MAT_DIR = Path("/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/hybrid_ac3d/Mat")
MAT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Phase frames for figure 3
TIME_FRAMES = [0, 20, 40, 60, 80, 100]  # indices (i.e., [0, 20dt, ..., 100dt])

# --------------------
# (NEW) Downsample knob for 3D volumes saved into MAT (1 = full resolution)
# --------------------
VOLUME_DOWNSAMPLE = 1


# --------------------
# Helpers (no code changes in your repo)
# --------------------
def _label_method(model_name: str, pde_weight: float):
    if pde_weight == 0.0:
        return 'FNO4d' if model_name == 'FNO4d' else 'MHNO'
    elif pde_weight == 1.0:
        return 'PurePhysics'
    else:
        return 'PENCO'


def _debug_terms(u_in_last, y_pred):
    dt, dx, eps2 = CFG.DT, CFG.DX, CFG.EPS2
    up = y_pred.squeeze(-1)
    u0 = u_in_last.squeeze(-1)
    ut = (up - u0) / dt
    mu = mu_ac(up, dx, eps2, dealias=True)
    debug_ut = ut.pow(2).mean()
    debug_mu = CFG.DEBUG_MU_SCALE * mu.pow(2).mean()
    return debug_ut, debug_mu


@torch.no_grad()
def _relative_l2_vs_time(model, mat_path, test_ids, T_in, Nt):
    import h5py
    relL2_t = []
    with h5py.File(mat_path, "r") as f:
        dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
        Nz, Ny, Nx, Nt_saved, Ns = dset.shape
        assert Nt_saved == CFG.SAVED_STEPS
        for t in range(Nt + 1):
            rels = []
            for sid in test_ids:
                raw = np.array(dset[:, :, :, :, int(sid)], dtype=np.float32)
                gt = np.transpose(raw, (3, 2, 1, 0))  # (Nt,Nx,Ny,Nz)
                # autoregressive prediction up to Nt
                # (cache per sid would be faster; kept simple)
                pred = rollout_autoregressive(model, gt, T_in, Nt=Nt)
                rel = np.linalg.norm(pred[t].ravel() - gt[t].ravel()) / (np.linalg.norm(gt[t].ravel()) + 1e-12)
                rels.append(rel)
            relL2_t.append(float(np.mean(rels)))
    return np.array(relL2_t, dtype=np.float32)  # shape (Nt+1,)


@torch.no_grad()
def _collect_phase_frames(model, mat_path, test_id, frames, T_in, Nt):
    import h5py
    with h5py.File(mat_path, "r") as f:
        raw = np.array(f["phi"][:, :, :, :, int(test_id)], dtype=np.float32)  # (Nz,Ny,Nx,Nt)
    gt = np.transpose(raw, (3, 2, 1, 0))  # (Nt,Nx,Ny,Nz)
    pred = rollout_autoregressive(model, gt, T_in, Nt=Nt)  # (Nt+1, S,S,S)
    S = gt.shape[1];
    zc = S // 2
    exact_slices = []
    pred_slices = []
    for t in frames:
        exact_slices.append(gt[t, :, :, zc])
        pred_slices.append(pred[t, :, :, zc])
    return np.stack(pred_slices, axis=0), np.stack(exact_slices, axis=0)  # (len(frames), S, S)


# --------------------
# (NEW) Collect full 3D volumes for selected time frames
# --------------------
@torch.no_grad()
def _collect_phase_volumes(model, mat_path, test_id, frames, T_in, Nt, downsample=1):
    """
    Collect full 3D volumes (Nx,Ny,Nz) for selected time frames.
    Optionally downsample isotropically by an integer factor to reduce MAT size.
    Returns:
        pred_vols:  (len(frames), S', S', S')
        exact_vols: (len(frames), S', S', S')
    """
    import h5py
    with h5py.File(mat_path, "r") as f:
        raw = np.array(f["phi"][:, :, :, :, int(test_id)], dtype=np.float32)  # (Nz,Ny,Nx,Nt)
    gt = np.transpose(raw, (3, 2, 1, 0))  # (Nt, Nx, Ny, Nz)
    pred = rollout_autoregressive(model, gt, T_in, Nt=Nt)  # (Nt+1, S, S, S)

    vols_exact, vols_pred = [], []
    for t in frames:
        v_gt = gt[t]
        v_pr = pred[t]
        if downsample > 1:
            v_gt = v_gt[::downsample, ::downsample, ::downsample]
            v_pr = v_pr[::downsample, ::downsample, ::downsample]
        vols_exact.append(v_gt)
        vols_pred.append(v_pr)

    return np.stack(vols_pred, axis=0), np.stack(vols_exact, axis=0)


def _init_model(model_name):
    if model_name == 'FNO4d':
        model = FNO4d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES, modes4_internal=None,
            width=CFG.WIDTH, width_q=CFG.WIDTH_Q, T_in_channels=CFG.T_IN_CHANNELS,
            n_layers=CFG.N_LAYERS
        ).to(CFG.DEVICE)
    else:
        model = TNO3d(
            modes1=CFG.MODES, modes2=CFG.MODES, modes3=CFG.MODES,
            width=CFG.WIDTH, width_q=CFG.WIDTH_Q, width_h=CFG.WIDTH_H,
            T_in=CFG.T_IN_CHANNELS, T_out=1, n_layers=CFG.N_LAYERS
        ).to(CFG.DEVICE)
    return model


def _train_one(model, train_loader, test_loader, pde_weight):
    # Mirror your train_fno_hybrid to LOG values (we’re not changing your code).
    optimizer = Adam(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)

    w_mid, w_fft, w_mm = 1.0, 1.0, 1.0
    MID_NORM_MIX = 0.5
    CLIP_NORM = 1.0

    logs = {
        'epoch': [],
        'data_loss': [], 'phys_loss': [], 'total_loss': [],
        'energy_loss': [], 'scheme_loss': [],
        'ut_mse': [], 'mu_mse': [],
        'test_relL2': [], 'lr': []
    }

    # print header exactly like utilities.py
    print(
        "Epoch |   Time   | DataLoss | PhysLoss | TotalLoss | u_t Term | μ_spatial Term | Test relL2 | energy_loss | scheme_loss | LR")

    for ep in range(CFG.EPOCHS):
        t1 = time.perf_counter()  # start timing this epoch

        model.train()
        data_loss_acc = phys_loss_acc = total_loss_acc = 0.0
        energy_loss_acc = scheme_loss_acc = 0.0
        ut_mse_acc = mu_mse_acc = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(CFG.DEVICE, non_blocking=True)
            y = y.to(CFG.DEVICE, non_blocking=True)
            u_in_last = x[..., -1:]

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)

            if pde_weight == 0:
                loss_data = 1.0 * F.mse_loss(y_pred, y)
            else:
                loss_data = 1e4 * F.mse_loss(y_pred, y)

            l_mid_plain = physics_residual_midpoint(u_in_last, y_pred)
            l_mid_norm, _, _ = physics_residual_normalized(u_in_last, y_pred)
            l_mid = (1.0 - MID_NORM_MIX) * l_mid_plain + MID_NORM_MIX * l_mid_norm

            l_fft = scheme_residual_fourier(u_in_last, y_pred)
            l_mm, _, _ = loss_mm_projection(u_in_last, y_pred)
            loss_phys = 1e-4 * (w_mid * l_mid + w_fft * l_fft + w_mm * l_mm)

            u_si = semi_implicit_step(u_in_last, CFG.DT, CFG.DX, CFG.EPS2)
            loss_scheme = F.mse_loss(y_pred, u_si)
            loss_energy = 0.05 * energy_penalty(u_in_last, y_pred, CFG.DX, CFG.EPS2)

            loss_total = (1.0 - pde_weight) * loss_data + pde_weight * (loss_phys + loss_scheme + loss_energy)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            scheduler.step()

            debug_ut, debug_mu = _debug_terms(u_in_last, y_pred)

            data_loss_acc += loss_data.item()
            phys_loss_acc += loss_phys.item()
            energy_loss_acc += loss_energy.item()
            scheme_loss_acc += loss_scheme.item()
            total_loss_acc += loss_total.item()
            ut_mse_acc += debug_ut.item()
            mu_mse_acc += debug_mu.item()
            n_batches += 1

        # Eval rel L2 on test loader (one-step)
        model.eval()
        with torch.no_grad():
            rels = []
            for x, y in test_loader:
                x = x.to(CFG.DEVICE, non_blocking=True)
                y = y.to(CFG.DEVICE, non_blocking=True)
                y_pred = model(x)
                rels.append(relative_l2(y_pred, y))
            test_rel = torch.cat(rels, dim=0).mean().item()
        t2 = time.perf_counter()  # end timing
        lr = optimizer.param_groups[0]['lr']

        # print exactly like utilities.py, but only every 5 epochs
        if ep % 5 == 0:
            print(f"{ep:5d} | {t2 - t1:7.3f} | "
                  f"{data_loss_acc / n_batches:8.3e} | {phys_loss_acc / n_batches:8.3e} | {total_loss_acc / n_batches:9.3e} | "
                  f"{(ut_mse_acc / n_batches):8.3e} | {(mu_mse_acc / n_batches):14.3e} | "
                  f"{test_rel:10.3e} |  {energy_loss_acc:10.3e} |  {scheme_loss_acc:10.3e} | {lr: .2e}")

        logs['epoch'].append(ep)
        logs['data_loss'].append(data_loss_acc / n_batches)
        logs['phys_loss'].append(phys_loss_acc / n_batches)
        logs['total_loss'].append(total_loss_acc / n_batches)
        logs['energy_loss'].append(energy_loss_acc / n_batches)
        logs['scheme_loss'].append(scheme_loss_acc / n_batches)
        logs['ut_mse'].append(ut_mse_acc / n_batches)
        logs['mu_mse'].append(mu_mse_acc / n_batches)
        logs['test_relL2'].append(test_rel)
        logs['lr'].append(optimizer.param_groups[0]['lr'])

    # Convert to numpy for .mat-friendly
    for k in logs:
        logs[k] = np.array(logs[k], dtype=np.float32)
    return logs


def main():
    # Fixed global config values are read from CFG
    print("Using device:", CFG.DEVICE)

    print("PDE_WEIGHTS:", PDE_WEIGHTS)
    print("MODELS:", MODELS)
    print("N_TRAINS:", N_TRAINS)

    # For data-efficiency comparisons we’ll keep the same seed per scenario
    random_seed = CFG.SEED
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

    # Reusable, scenario-level results bucket
    scenarios = []

    # Dummy loaders call (to verify paths exist early)
    # (We’ll rebuild loaders for each N_TRAIN below)
    _ = CFG.MAT_DATA_PATH

    for ntrain in N_TRAINS:
        # update config N_TRAIN and dependent N_TEST
        CFG.N_TRAIN = int(ntrain)
        CFG.N_TEST = max(1, CFG.N_TRAIN // 4)

        # Build loaders ONCE per N_TRAIN (same split for all pde/model pairs)
        train_loader, test_loader, test_ids, normalizers = build_loaders()
        # Use the FIRST test id for phase-evolution snapshots
        rep_test_id = int(test_ids[0])

        for model_name in MODELS:
            for pw in PDE_WEIGHTS:
                # Init fresh model
                model = _init_model(model_name)

                # Train with logging
                logs = _train_one(model, train_loader, test_loader, pde_weight=pw)

                # Save model checkpoint
                method_label = _label_method(model_name, pw)
                ckpt_name = f"{model_name}_{method_label}_N{CFG.N_TRAIN}_pw{pw:.2f}_E{CFG.EPOCHS}.pt"
                ckpt_path = MODELS_DIR / ckpt_name
                torch.save({'state_dict': model.state_dict(),
                            'model': model_name,
                            'method': method_label,
                            'pde_weight': float(pw),
                            'N_TRAIN': int(CFG.N_TRAIN),
                            'EPOCHS': int(CFG.EPOCHS),
                            'config': {
                                'GRID_RESOLUTION': CFG.GRID_RESOLUTION,
                                'DT': CFG.DT,
                                'TOTAL_TIME_STEPS': CFG.TOTAL_TIME_STEPS,
                                'T_IN_CHANNELS': CFG.T_IN_CHANNELS,
                            }}, ckpt_path)

                # Relative L2 vs time (rollout on test set)
                relL2_vs_time = _relative_l2_vs_time(
                    model, CFG.MAT_DATA_PATH, test_ids,
                    T_in=CFG.T_IN_CHANNELS, Nt=CFG.TOTAL_TIME_STEPS
                )  # (Nt+1,)

                # Phase evolution central-slice frames (pred/exact) for representative sample
                pred_slices, exact_slices = _collect_phase_frames(
                    model, CFG.MAT_DATA_PATH, rep_test_id, TIME_FRAMES,
                    T_in=CFG.T_IN_CHANNELS, Nt=CFG.TOTAL_TIME_STEPS
                )  # shapes (len(frames), S, S)

                # --------------------
                # (NEW) Phase evolution full 3D volumes (pred/exact) for the same representative sample
                # --------------------
                pred_vols, exact_vols = _collect_phase_volumes(
                    model, CFG.MAT_DATA_PATH, rep_test_id, TIME_FRAMES,
                    T_in=CFG.T_IN_CHANNELS, Nt=CFG.TOTAL_TIME_STEPS,
                    downsample=VOLUME_DOWNSAMPLE
                )  # shapes (len(frames), S', S', S')

                # Pack scenario result
                scenarios.append({
                    'model': model_name,
                    'method_label': method_label,  # 'FNO4d', 'MHNO', 'PENCO', 'PurePhysics'
                    'pde_weight': float(pw),
                    'N_TRAIN': int(CFG.N_TRAIN),
                    'epochs': int(CFG.EPOCHS),
                    # Per-epoch logs:
                    'data_loss': logs['data_loss'],
                    'phys_loss': logs['phys_loss'],
                    'energy_loss': logs['energy_loss'],
                    'scheme_loss': logs['scheme_loss'],
                    'total_loss': logs['total_loss'],
                    'ut_mse': logs['ut_mse'],
                    'mu_mse': logs['mu_mse'],
                    'test_relL2': logs['test_relL2'],
                    # Rollout metric:
                    'relL2_vs_time': relL2_vs_time,  # length Nt+1
                    # Phase evolution slices:
                    'phase_pred_slices': pred_slices,  # (len(frames), S, S)
                    'phase_exact_slices': exact_slices,  # (len(frames), S, S)
                    'time_frames_idx': np.array(TIME_FRAMES, dtype=np.int32),
                    # --------------------
                    # (NEW) Phase evolution volumes:
                    # --------------------
                    'phase_pred_volumes': pred_vols,  # (len(frames), S', S', S')
                    'phase_exact_volumes': exact_vols,  # (len(frames), S', S', S')
                    'volume_downsample': int(VOLUME_DOWNSAMPLE),
                    'rep_test_id': int(rep_test_id),
                    # Coefficients used in the loss (for reproducibility/table):
                    'coefficients': {
                        'w_mid': 1.0, 'w_fft': 1.0, 'w_mm': 1.0,
                        'MID_NORM_MIX': 0.5, 'CLIP_NORM': 1.0,
                        'data_scale_if_hybrid': 1e4, 'phys_scale': 1e-4,
                        'energy_weight': 0.05
                    },
                    # Saved model path:
                    'model_path': str(ckpt_path)
                })

                # Free CUDA memory between scenarios
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Assemble global structure for MATLAB
    out = {
        'meta': {
            'GRID_RESOLUTION': CFG.GRID_RESOLUTION,
            'DX': CFG.DX,
            'DT': CFG.DT,
            'TOTAL_TIME_STEPS': CFG.TOTAL_TIME_STEPS,
            'SAVED_STEPS': CFG.SAVED_STEPS,
            'EPSILON_PARAM': CFG.EPSILON_PARAM,
            'MAT_DATA_PATH': CFG.MAT_DATA_PATH,
            'EPOCHS': CFG.EPOCHS,
            'T_IN_CHANNELS': CFG.T_IN_CHANNELS,
            'TIME_FRAMES_IDX': np.array(TIME_FRAMES, dtype=np.int32),
            'PDE_WEIGHTS': np.array(PDE_WEIGHTS, dtype=np.float32),
            'MODELS': np.array(MODELS, dtype=object),
            'N_TRAINS': np.array(N_TRAINS, dtype=np.int32),
            # (NEW) Expose the downsample used for saved volumes
            'VOLUME_DOWNSAMPLE': int(VOLUME_DOWNSAMPLE),
        },
        # Convert list[dict] to MATLAB-compatible struct array
        'scenarios': np.array(scenarios, dtype=object)
    }

    # Save .mat
    mat_name = f"AC3D_sweep_results_{int(time.time())}.mat"
    mat_path = MAT_DIR / mat_name
    savemat(str(mat_path), out, do_compression=True)
    print(f"\nSaved results MAT to: {mat_path}")
    print(f"Models saved under:   {MODELS_DIR}")



if __name__ == "__main__":
    main()
