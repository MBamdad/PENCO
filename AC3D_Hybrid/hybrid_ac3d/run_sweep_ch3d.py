# main_sweep_ch3d.py
# CH3D sweep that preserves training/eval math IDENTICAL to utilities (CH path),
# and adds logging + .mat/model saving (same policy as SH/MBE sweeps).
#
# Scenarios policy:
#   • Fixed test split (utilities.build_loaders + N_TEST_FIXED)
#   • Step-based training (per-batch LR stepping) with STEPS_PER_EPOCH_EFF (fallback=20)
#   • Reseed before every scenario

import os, time, random, math
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
    build_loaders,
    relative_l2,
    rollout_autoregressive,
)
from functions import (
    # core residuals / penalties
    scheme_residual_fourier,
    energy_penalty,               # (AC/CH energy hinge)
    # ---- CH-specific (mirror SH/PFC/MBE structure) ----
    semi_implicit_step_ch,        # public CH teacher
    physics_collocation_tau_L2_CH,# CH collocation residual (L2)
    physics_guided_update_ch_optimal,
    low_k_mse,
    mass_project_pred,            # hard mass projection (CH invariance)
)

# --------------------
# Sweep parameters
# --------------------
DEFAULT_PDE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_MODELS      = ['FNO4d', 'TNO3d']   # TNO3d labeled "MHNO" in saved results
DEFAULT_N_TRAINS    = [50, 100, 200]

# Phase frames for figures
TIME_FRAMES = [0, 20, 40, 60, 80, 100]  # indices
VOLUME_DOWNSAMPLE = 1  # set >1 to shrink 3D volumes in the mat for speed/size

# Save dirs (CH3D)
MODELS_DIR = Path("./CH3d_models")
MAT_DIR    = Path("./CH3d_mat")
FAST_SAVE  = True  # for savemat(compression=...) choice

# --------------------
# Helpers (match run_sweep style)
# --------------------
def _asF32F(arr):
    return np.asfortranarray(np.array(arr, dtype=np.float32, copy=False))

def _label_method(model_name: str, pde_weight: float):
    if pde_weight == 0.0:
        return 'FNO4d' if model_name == 'FNO4d' else 'MHNO'
    elif pde_weight == 1.0:
        return 'PurePhysics'
    else:
        return 'PENCO'

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

def _relative_l2_vs_time(model, mat_path, test_ids, T_in, Nt):
    import h5py
    rel_sum = np.zeros(Nt + 1, dtype=np.float64)
    cnt = 0
    with h5py.File(mat_path, "r") as f:
        dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
        for sid in test_ids:
            raw = np.array(dset[:, :, :, :, int(sid)], dtype=np.float32)
            gt  = np.transpose(raw, (3, 2, 1, 0))  # (Nt,S,S,S)
            pred = rollout_autoregressive(model, gt, T_in, Nt=Nt)
            for t in range(Nt + 1):
                num = np.linalg.norm(pred[t].ravel() - gt[t].ravel())
                den = np.linalg.norm(gt[t].ravel()) + 1e-12
                rel_sum[t] += (num / den)
            cnt += 1
    rel = rel_sum / max(cnt, 1)
    return _asF32F(rel)

def _relative_l2_vs_time_stats(model, mat_path, test_ids, T_in, Nt):
    """
    Returns per-time-step statistics of relative L2 error across the test set:
    mean, std, q1, median, q3, plus the time-step index vector [0..Nt].
    """
    import h5py
    per_sample = []
    with h5py.File(mat_path, "r") as f:
        dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
        for sid in test_ids:
            raw = np.array(dset[:, :, :, :, int(sid)], dtype=np.float32)
            gt  = np.transpose(raw, (3, 2, 1, 0))                  # (Nt,S,S,S)
            pred = rollout_autoregressive(model, gt, T_in, Nt=Nt)  # (Nt,S,S,S)
            e = np.empty(Nt + 1, dtype=np.float64)
            for t in range(Nt + 1):
                num = np.linalg.norm(pred[t].ravel() - gt[t].ravel())
                den = np.linalg.norm(gt[t].ravel()) + 1e-12
                e[t] = num / den
            per_sample.append(e)

    if len(per_sample) == 0:
        per_sample = np.zeros((1, Nt + 1), dtype=np.float64)
    else:
        per_sample = np.stack(per_sample, axis=0)  # (Ns, Nt+1)

    stats = {
        'time_steps': np.arange(Nt + 1, dtype=np.int32),
        'mean':   per_sample.mean(axis=0).astype(np.float32),
        'std':    per_sample.std(axis=0, ddof=0).astype(np.float32),
        'q1':     np.quantile(per_sample, 0.25, axis=0).astype(np.float32),
        'median': np.quantile(per_sample, 0.50, axis=0).astype(np.float32),
        'q3':     np.quantile(per_sample, 0.75, axis=0).astype(np.float32),
    }
    return stats

def _collect_frames_and_volumes(model, mat_path, test_id, frames, T_in, Nt, downsample=1):
    import h5py
    with h5py.File(mat_path, "r") as f:
        raw = np.array(f["phi"][:, :, :, :, int(test_id)], dtype=np.float32)
    gt  = np.transpose(raw, (3, 2, 1, 0))      # (Nt, S,S,S)
    pred = rollout_autoregressive(model, gt, T_in, Nt=Nt)

    S = gt.shape[1]; zc = S // 2
    exact_slices, pred_slices = [], []
    exact_vols,   pred_vols   = [], []

    for t in frames:
        exact_slices.append(gt[t, :, :, zc])
        pred_slices.append(pred[t, :, :, zc])

        v_gt = gt[t]; v_pr = pred[t]
        if downsample > 1:
            v_gt = v_gt[::downsample, ::downsample, ::downsample]
            v_pr = v_pr[::downsample, ::downsample, ::downsample]
        exact_vols.append(v_gt); pred_vols.append(v_pr)

    return (
        _asF32F(np.stack(pred_slices, axis=0)),  # (len(frames), S, S)
        _asF32F(np.stack(exact_slices, axis=0)),
        _asF32F(np.stack(pred_vols,   axis=0)),  # (len(frames), S',S',S')
        _asF32F(np.stack(exact_vols,  axis=0))
    )

# --------------------
# TRAIN LOOP: CH path with step-based training + logging
# (math identical to utilities CH branch)
# --------------------
def train_fno_hybrid_LOGGING_STEPBASED(model, train_loader, test_loader, optimizer, scheduler, device, pde_weight=None):
    pde_weight = CFG.PDE_WEIGHT if pde_weight is None else pde_weight
    assert CFG.PROBLEM == 'CH3D', "This sweep is intended for CH3D."

    CLIP_NORM = 1.0
    logs = {k: [] for k in [
        'epoch', 'data_loss', 'phys_loss', 'energy_loss', 'scheme_loss',
        'total_loss', 'test_relL2', 'l_mid_norm_ch', 'lr'
    ]}

    from itertools import cycle
    from timeit import default_timer

    steps_per_epoch = int(getattr(CFG, "STEPS_PER_EPOCH_EFF",
                           getattr(CFG, "STEPS_PER_EPOCH", 20)))
    train_iter = cycle(train_loader)

    print("Epoch |   Time   | DataLoss | PhysLoss | TotalLoss | Test relL2 | energy_loss | scheme_loss | l_mid_norm_ch | LR")

    for ep in range(CFG.EPOCHS):
        model.train()
        t1 = default_timer()
        data_loss_acc = phys_loss_acc = total_loss_acc = l_mid_norm_acc = 0.0
        energy_loss_acc = scheme_loss_acc = 0.0
        n_batches = 0

        for _ in range(steps_per_epoch):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)

            optimizer.zero_grad(set_to_none=True)

            # forward
            y_pred = model(x)

            # data term (same scaling style as utilities)
            loss_data = (1.0 if pde_weight == 0 else 1e3) * F.mse_loss(y_pred, y)

            # ----- CH physics bundle (IDENTICAL to utilities CH branch) -----
            # gentle ramps like SH/PFC/MBE
            epoch_frac = ep / max(1, (CFG.EPOCHS - 1))
            w_scheme = 0.32 - 0.12 * epoch_frac
            w_lowk = 0.25 + 0.60 * (epoch_frac ** 2)

            # --- forward (CH: hard mass projection to enforce invariance) ---
            y_pred = model(x)
            y_hat = mass_project_pred(y_pred, u_in_last)

            # --- Fourier preconditioned semi-implicit residual (CH-aware) ---
            l_fft = scheme_residual_fourier(u_in_last, y_hat)

            # --- L2 Gauss–Lobatto collocation (same nodes as SH/PFC/MBE) ---
            tau_off = 1.0 / (2.0 * math.sqrt(5.0))
            l_tau1 = physics_collocation_tau_L2_CH(u_in_last, y_hat, tau=(0.5 - tau_off))
            l_tau2 = physics_collocation_tau_L2_CH(u_in_last, y_hat, tau=(0.5 + tau_off))
            l_mid_norm = 0.5 * (l_tau1 + l_tau2)

            # --- teacher consistency (CH semi-implicit), with PGU on step-2 ---
            u_si1 = semi_implicit_step_ch(u_in_last, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)
            loss_scheme1 = F.mse_loss(y_hat, u_si1)

            with torch.no_grad():
                u_si2 = semi_implicit_step_ch(u_si1, CFG.DT, CFG.DX, CFG.EPSILON_PARAM)

            x2 = torch.cat([x[..., 1:], y_hat], dim=-1)
            y_hat2 = model(x2)
            # y_hat2 = physics_guided_update_ch_optimal(
            #    x2[..., -1:], y_hat2, alpha_cap=0.6, low_k_snap_frac=0.45
            # )
            loss_scheme2 = F.mse_loss(y_hat2, u_si2)
            loss_scheme = w_scheme * (0.6 * loss_scheme1 + 0.4 * loss_scheme2)

            # --- spectral low-k anchor (a bit stronger for CH) ---
            l_lowk = low_k_mse(y_hat, u_si1, frac=0.50)

            # --- physics mix (same base scale), add tiny H^{-1} terms ---
            loss_phys = 8e-3 * (l_fft + 0.6 * l_mid_norm + w_lowk * 0.70 * l_lowk)
            # --- energy hinge (AC/CH) + very soft mass regularizer ---
            loss_energy = 0.03 * energy_penalty(u_in_last, y_hat, CFG.DX, CFG.EPS2)

            # total
            loss_total = (1.0 - pde_weight) * loss_data + pde_weight * (loss_phys + loss_scheme + loss_energy)

            # backward + step
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            scheduler.step()  # per-batch LR step (step-based training)

            # accumulators
            data_loss_acc   += loss_data.item()
            phys_loss_acc   += loss_phys.item()
            l_mid_norm_acc  += l_mid_norm.item()
            energy_loss_acc += loss_energy.item()
            scheme_loss_acc += loss_scheme.item()
            total_loss_acc  += loss_total.item()
            n_batches       += 1

        # eval (same as utilities path)
        model.eval()
        with torch.no_grad():
            rels = []
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_pred = model(x)
                rels.append(relative_l2(y_pred, y))
            test_rel = torch.cat(rels, dim=0).mean().item()

        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']

        # epoch print
        if ep % 15 == 0:
            print(f"{ep:5d} | {t2-t1:7.3f} | "
                  f"{data_loss_acc/n_batches:8.3e} | {phys_loss_acc/n_batches:8.3e} | {total_loss_acc/n_batches:9.3e} | "
                  f"{test_rel:10.3e} |  {energy_loss_acc/n_batches:10.3e} |  {scheme_loss_acc/n_batches:10.3e} | {l_mid_norm_acc:10.3e} | {lr: .2e}")

        # logs
        logs['epoch'].append(ep)
        logs['data_loss'].append(data_loss_acc/n_batches)
        logs['phys_loss'].append(phys_loss_acc/n_batches)
        logs['energy_loss'].append(energy_loss_acc/n_batches)
        logs['scheme_loss'].append(scheme_loss_acc/n_batches)
        logs['total_loss'].append(total_loss_acc/n_batches)
        logs['test_relL2'].append(test_rel)
        logs['l_mid_norm_ch'].append(l_mid_norm_acc/n_batches)
        logs['lr'].append(lr)

    for k in logs:
        logs[k] = _asF32F(np.array(logs[k], dtype=np.float32))
    return logs


def set_seeds(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    assert CFG.PROBLEM == 'CH3D', f"config.PROBLEM must be 'CH3D' (got {CFG.PROBLEM})"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MAT_DIR.mkdir(parents=True, exist_ok=True)

    print("Using device:", CFG.DEVICE)
    print("TIME_FRAMES:", TIME_FRAMES)
    print("STEPS_PER_EPOCH:", CFG.STEPS_PER_EPOCH)

    set_seeds(CFG.SEED)

    scenarios = []

    for ntrain in DEFAULT_N_TRAINS:
        CFG.N_TRAIN = int(ntrain)
        # Fixed test split handled in build_loaders via N_TEST_FIXED

        for model_name in DEFAULT_MODELS:
            CFG.MODEL = model_name
            for pw in DEFAULT_PDE_WEIGHTS:
                CFG.PDE_WEIGHT = float(pw)

                # --- Reseed so each scenario starts from the same RNG state ---
                set_seeds(CFG.SEED)

                # Build loaders AFTER setting PDE_WEIGHT (PURE_PHYSICS_USE_ALL may apply)
                train_loader, test_loader, test_ids, _ = build_loaders()

                # ===== Step-based scaling (identical to other sweeps & single-run) =====
                base_steps = int(getattr(CFG, "STEPS_PER_EPOCH", 20))
                if (CFG.PDE_WEIGHT < 1.0) and getattr(CFG, "SCALE_STEPS_WITH_NTRAIN", False):
                    N_ref = max(1, int(getattr(CFG, "N_TRAIN_REF", 50)))
                    N_cur = max(1, int(getattr(CFG, "N_TRAIN_ACTUAL",
                                               getattr(CFG, "N_TRAIN", N_ref))))
                    STEPS_PER_EPOCH_EFF = max(1, int(round(base_steps * N_cur / N_ref)))
                else:
                    STEPS_PER_EPOCH_EFF = base_steps

                setattr(CFG, "STEPS_PER_EPOCH_EFF", STEPS_PER_EPOCH_EFF)
                #total_steps = CFG.EPOCHS * STEPS_PER_EPOCH_EFF
                total_steps = CFG.N_TRAIN_REF * STEPS_PER_EPOCH_EFF
                print(f"[Budget] N_TRAIN={CFG.N_TRAIN} (actual={getattr(CFG,'N_TRAIN_ACTUAL',CFG.N_TRAIN)}), "
                      f"steps/epoch={STEPS_PER_EPOCH_EFF}, total={total_steps}")
                # ===================================================================

                rep_test_id = int(test_ids[0])

                # ---- model, optim, sched ----
                model = _init_model(model_name)
                optimizer = Adam(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
                scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)  # per-batch schedule

                # ---- TRAIN (step-based) + logs ----
                logs = train_fno_hybrid_LOGGING_STEPBASED(
                    model, train_loader, test_loader, optimizer, scheduler, CFG.DEVICE, pde_weight=CFG.PDE_WEIGHT
                )

                # ---- METRICS & FRAMES ----
                relL2_vs_time = _relative_l2_vs_time(
                    model, CFG.MAT_DATA_PATH, test_ids,
                    T_in=CFG.T_IN_CHANNELS, Nt=CFG.TOTAL_TIME_STEPS
                )
                pred_slices, exact_slices, pred_vols, exact_vols = _collect_frames_and_volumes(
                    model, CFG.MAT_DATA_PATH, rep_test_id, TIME_FRAMES,
                    T_in=CFG.T_IN_CHANNELS, Nt=CFG.TOTAL_TIME_STEPS, downsample=VOLUME_DOWNSAMPLE
                )

                # ---- SAVE MODEL (CPU checkpoint) ----
                method_label = _label_method(model_name, pw)
                ckpt_name = f"{model_name}_{method_label}_N{CFG.N_TRAIN}_pw{pw:.2f}_E{CFG.EPOCHS}.pt"
                ckpt_path = MODELS_DIR / ckpt_name
                model_cpu = model.to('cpu')
                torch.save({
                    'state_dict': model_cpu.state_dict(),
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
                    }
                }, ckpt_path)

                # ---- per-time-step error stats ----
                rel_stats = _relative_l2_vs_time_stats(
                    model, CFG.MAT_DATA_PATH, test_ids,
                    T_in=CFG.T_IN_CHANNELS, Nt=CFG.TOTAL_TIME_STEPS
                )

                # ---- PACK SCENARIO ----
                scenarios.append({
                    'model': model_name,
                    'method_label': method_label,
                    'pde_weight': float(pw),
                    'N_TRAIN': int(CFG.N_TRAIN),
                    'epochs': int(CFG.EPOCHS),

                    # training curves
                    'data_loss': logs['data_loss'],
                    'phys_loss': logs['phys_loss'],
                    'energy_loss': logs['energy_loss'],
                    'scheme_loss': logs['scheme_loss'],
                    'total_loss': logs['total_loss'],
                    'test_relL2': logs['test_relL2'],

                    # mean rel. L2 vs time
                    'relL2_vs_time': relL2_vs_time,

                    # per-time-step error stats
                    'relL2_stats_time_steps': _asF32F(np.array(rel_stats['time_steps'], dtype=np.int32)),
                    'relL2_stats_mean': _asF32F(rel_stats['mean']),
                    'relL2_stats_std': _asF32F(rel_stats['std']),
                    'relL2_stats_q1': _asF32F(rel_stats['q1']),
                    'relL2_stats_median': _asF32F(rel_stats['median']),
                    'relL2_stats_q3': _asF32F(rel_stats['q3']),

                    # frames & volumes
                    'phase_pred_slices': pred_slices,
                    'phase_exact_slices': exact_slices,
                    'time_frames_idx': _asF32F(np.array(TIME_FRAMES, dtype=np.int32)),
                    'phase_pred_volumes': pred_vols,
                    'phase_exact_volumes': exact_vols,
                    'volume_downsample': int(VOLUME_DOWNSAMPLE),
                    'rep_test_id': int(rep_test_id),
                    'model_path': str(ckpt_path)
                })

                # free GPU
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- SAVE one MATLAB file ----
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
            'TIME_FRAMES_IDX': _asF32F(np.array(TIME_FRAMES, dtype=np.int32)),
            'PDE_WEIGHTS': _asF32F(np.array(DEFAULT_PDE_WEIGHTS, dtype=np.float32)),
            'MODELS': np.array(DEFAULT_MODELS, dtype=object),
            'N_TRAINS': _asF32F(np.array(DEFAULT_N_TRAINS, dtype=np.int32)),
            'VOLUME_DOWNSAMPLE': int(VOLUME_DOWNSAMPLE),

            # record budgeting knobs
            'STEPS_PER_EPOCH': int(getattr(CFG, "STEPS_PER_EPOCH", 20)),
            'N_TEST_FIXED': int(getattr(CFG, "N_TEST_FIXED", 40)),
            'SCALE_STEPS_WITH_NTRAIN': bool(getattr(CFG, "SCALE_STEPS_WITH_NTRAIN", False)),
            'N_TRAIN_REF': int(getattr(CFG, "N_TRAIN_REF", 50)),
        },
        'scenarios': np.array(scenarios, dtype=object)
    }

    mat_name = f"CH3D_sweep_results_{int(time.time())}.mat"
    mat_path = MAT_DIR / mat_name
    savemat(str(mat_path), out, do_compression=not FAST_SAVE)
    print(f"\nSaved results MAT to: {mat_path.resolve()}")
    print(f"Models saved under:   {MODELS_DIR.resolve()}")

if __name__ == "__main__":
    main()
