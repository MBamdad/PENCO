import h5py, numpy as np, torch, random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import config
from itertools import cycle
from torch.cuda.amp import autocast

import math
from functions import (
    physics_residual_midpoint, scheme_residual_fourier, loss_mm_projection,
    semi_implicit_step, energy_penalty, physics_residual_normalized,physics_collocation_GL3_AC,energy_dissipation_identity_AC,
    hminus1_mse  ,physics_collocation_GL4_AC,physics_collocation_Radau3R_AC,residual_orthogonality_P2_AC,weak_residual_lowK_AC,mean_evolution_AC,
    ac_scheme_residual_fourier,physics_collocation_GL3_CH_Hm1,
    # ---- SH additions ----
    semi_implicit_step_sh,                    # SH semi-implicit teacher step
    physics_collocation_tau_L2_SH,            # SH collocation residual (L2)
    energy_penalty_sh,                        # SH energy hinge
    _mid_residual_norm_sh, physics_guided_update_sh_optimal,ch_two_step_consistency,
    low_k_mse,
    # ---- PFC additions ----
    semi_implicit_step_pfc,
    physics_collocation_tau_L2_PFC,
    physics_guided_update_pfc_optimal, energy_penalty_pfc,
    semi_implicit_step_mbe,
    physics_collocation_tau_L2_MBE, mass_project_pred,
    energy_penalty_mbe, physics_guided_update_mbe_optimal,mass_penalty, physics_collocation_tau_Hm1_MBE,
    # ---- NEW: identical-form L2 collocation for AC & CH ----
    physics_collocation_tau_L2_AC,physics_collocation_GL3_AC_interface,ac_mm_prox_step, loss_ac_mm_prox,
    pde_rhs,
    # ---- CH additions ----
    semi_implicit_step_ch,physics_collocation_GL5_CH_Hm2,
    physics_collocation_tau_L2_CH,physics_guided_update_ch_optimal,physics_collocation_tau_Hm1_CH,
    physics_collocation_tau_L2_AC,physics_guided_update_ac_optimal,
    physics_collocation_GL3_CH,energy_dissipation_identity_CH,ch_symbolic_residual_Hm2,
    physics_collocation_GL5_SH_Hm2,energy_dissipation_identity_SH
)


import matplotlib
matplotlib.use('TkAgg')

with h5py.File(config.MAT_DATA_PATH, "r") as f:
    print(config.PROBLEM, f["phi"].shape)

# ---------------------
# Dataset: load chosen trajectories into RAM ONCE
# ---------------------
class AC3DTrajectoryDataset(Dataset):
    """
    Holds full trajectories for selected sample_ids in RAM.
    __getitem__ returns the trajectory tensor (Nt, S, S, S).
    """
    def __init__(self, mat_path, sample_ids, dtype=np.float32):
        super().__init__()
        self.sample_ids = np.array(sample_ids)
        with h5py.File(mat_path, "r") as f:
            dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
            Nz, Ny, Nx, Nt, Ns = dset.shape
            self.Nz, self.Ny, self.Nx, self.Nt = Nz, Ny, Nx, Nt
            self.data = []
            for sid in self.sample_ids:
                raw = np.array(dset[:, :, :, :, sid], dtype=dtype)   # (Nz,Ny,Nx,Nt)
                traj = np.transpose(raw, (3,2,1,0))                  # (Nt,Nx,Ny,Nz)
                self.data.append(traj)
            self.data = np.stack(self.data, axis=0)                  # (Ns_sel,Nt,Nx,Ny,Nz)

        X = self.data
        self._mean = float(X.mean()); self._std = float(X.std() + 1e-8)
        class _Norm:
            def __init__(self, m, s): self.m, self.s = m, s
            def encode(self, t): return (t - self.m)/self.s
            def decode(self, t): return t*self.s + self.m
        self.normalizer_x = _Norm(self._mean, self._std)
        self.normalizer_y = _Norm(self._mean, self._std)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.from_numpy(self.data[idx])  # (Nt,S,S,S)

# ---------------------
# Collate: convert trajectories -> (x,y) windows
# ---------------------
def make_windowed_collate(T_in=4, t_min=None, t_max=None, normalized=False, normalizers=None):
    y_norm = normalizers[1] if (normalized and normalizers is not None) else None
    def _collate(batch):
        Nt = batch[0].shape[0]
        t0 = T_in - 1 if t_min is None else max(T_in-1, t_min)
        t1 = Nt - 2   if t_max is None else min(Nt-2,   t_max)

        xs, ys = [], []
        for traj in batch:                 # traj: (Nt,S,S,S)
            t = random.randint(t0, t1)
            x = traj[t-(T_in-1):t+1]       # (T_in,S,S,S)
            y = traj[t+1]                  # (S,S,S)
            x = x.permute(1,2,3,0).contiguous()   # (S,S,S,T_in)
            y = y.unsqueeze(-1).contiguous()      # (S,S,S,1)
            xs.append(x); ys.append(y)
        x = torch.stack(xs, dim=0)         # (B,S,S,S,T_in)
        y = torch.stack(ys, dim=0)         # (B,S,S,S,1)
        if y_norm is not None:
            x = (x - y_norm.m)/y_norm.s
            y = (y - y_norm.m)/y_norm.s
        return x, y
    return _collate

# ---------------------
# Loaders (your preferred split API)
# ---------------------


def build_loaders():
    rng = np.random.default_rng(config.SEED)

    # --- read Ns from the file (phi: (Nz,Ny,Nx,Nt,Ns)) ---
    with h5py.File(config.MAT_DATA_PATH, "r") as f:
        Ns = int(f["phi"].shape[-1])

    # deterministically shuffle IDs once
    all_ids = np.arange(Ns)
    rng.shuffle(all_ids)

    # ----- FIX 1: make test set independent of N_TRAIN -----
    n_test = min(int(getattr(config, "N_TEST_FIXED", 100)), Ns - 1)  # keep at least 1 train
    test_ids = all_ids[:n_test]
    train_pool = all_ids[n_test:]

    # ----- FIX 2: in pure-physics mode, ignore N_TRAIN and use full pool -----
    use_all = bool(getattr(config, "PURE_PHYSICS_USE_ALL", True)) and (config.PDE_WEIGHT == 1.0)
    if use_all:
        chosen_train_ids = train_pool
    else:
        n_train_req = int(getattr(config, "N_TRAIN", len(train_pool)))
        n_train = max(1, min(n_train_req, len(train_pool)))
        chosen_train_ids = train_pool[:n_train]

    # build RAM datasets over the chosen IDs
    train_base = AC3DTrajectoryDataset(config.MAT_DATA_PATH, chosen_train_ids)
    test_base  = AC3DTrajectoryDataset(config.MAT_DATA_PATH, test_ids)

    normalizers = [train_base.normalizer_x, train_base.normalizer_y]

    collate = make_windowed_collate(
        T_in=config.T_IN_CHANNELS, t_min=0, t_max=config.TOTAL_TIME_STEPS-1,
        normalized=False, normalizers=normalizers
    )

    train_loader = DataLoader(train_base, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=collate)
    test_loader = DataLoader(test_base, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True,
                             collate_fn=collate)

    # deterministic list of test IDs (now fixed!)
    test_indices = list(test_ids)
    setattr(config, "N_TRAIN_ACTUAL", len(chosen_train_ids))
    print(f"[Split] train_ids={len(chosen_train_ids)}, test_ids={len(test_ids)}")

    return train_loader, test_loader, test_indices, normalizers


# ---------------------
# Utilities
# ---------------------
def relative_l2(a, b, eps=1e-12):
    diff = (a - b).flatten(start_dim=1)
    denom = b.flatten(start_dim=1)
    num = torch.sqrt(torch.sum(diff**2, dim=1) + eps)
    den = torch.sqrt(torch.sum(denom**2, dim=1) + eps)
    return (num / den)  # (B,)

# ---------------------
# Training: hybrid
# ---------------------


########################################
#########################################

## Correct
def train_fno_hybrid(model, train_loader, test_loader, optimizer, scheduler, device, pde_weight=None):
    pde_weight = config.PDE_WEIGHT if pde_weight is None else pde_weight

    # physics term weights (baseline-compatible)
    USE_AC = (config.PROBLEM == 'AC3D')  # NEW
    USE_CH = (config.PROBLEM == 'CH3D')
    USE_SH = (config.PROBLEM == 'SH3D')  # <-- NEW
    USE_PFC = (config.PROBLEM == 'PFC3D')  # NEW
    USE_MBE = (config.PROBLEM == 'MBE3D')  # <-- add this
    CLIP_NORM = 1.0


    print("Epoch |   Time   | DataLoss | PhysLoss | TotalLoss | Test relL2 | energy_loss | scheme_loss | l_mid_norm_ch_cc | LR")

    # FIXED number of updates per epoch (independent of N_TRAIN)
    steps_per_epoch = getattr(config, "STEPS_PER_EPOCH_EFF", getattr(config, "STEPS_PER_EPOCH", 20))


    train_iter = cycle(train_loader)  # infinite stream of batches

    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        data_loss_acc = phys_loss_acc = total_loss_acc = l_mid_norm_ch_cc = 0.0
        energy_loss_acc = scheme_loss_acc = 0.0
        ut_mse_acc = mu_mse_acc = 0.0
        n_batches = 0
        for _ in range(steps_per_epoch):  # ← fixed number of updates each epoch
            x, y = next(train_iter)  # ← independent of dataset length
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)
            optimizer.zero_grad(set_to_none=True)
            # forward
            y_pred = model(x)
            y_hat = y_pred
            # ---- data term (keep your scaling so behavior remains stable) ----
            if pde_weight == 0 or pde_weight == 1:
                loss_data =  F.mse_loss(y_pred, y)
            else:
                loss_data = 1e2 * F.mse_loss(y_pred, y)

            # Correct
            if USE_SH:
                # ----- SH physics bundle (matches utilities SH branch) -----
                # ramps
                epoch_frac = ep / max(1, (config.EPOCHS - 1))  # 0..1
                w_scheme = 0.32 - 0.12 * epoch_frac
                w_lowk = 0.25 + 0.60 * (epoch_frac ** 2)

                # residuals
                #l_fft = scheme_residual_fourier(u_in_last, y_hat)
                tau_off = 1.0 / (2.0 * math.sqrt(5.0))
                l_tau1 = physics_collocation_tau_L2_SH(u_in_last, y_hat, tau=(0.5 - tau_off))
                l_tau2 = physics_collocation_tau_L2_SH(u_in_last, y_hat, tau=(0.5 + tau_off))
                l_mid_norm = 0.5 * (l_tau1 + l_tau2)

                # teacher + gentle second step
                u_si1 = semi_implicit_step_sh(u_in_last, config.DT, config.DX, config.EPSILON_PARAM)
                loss_scheme1 = F.mse_loss(y_hat, u_si1)
                #with torch.no_grad():
                #    u_si2 = semi_implicit_step_sh(u_si1, config.DT, config.DX, config.EPSILON_PARAM)
                #x2 = torch.cat([x[..., 1:], y_hat], dim=-1)
                #y_hat2 = model(x2)
                #loss_scheme2 = F.mse_loss(y_hat2, u_si2)
                #loss_scheme = w_scheme * (0.6 * loss_scheme1 + 0.4 * loss_scheme2)
                loss_scheme = w_scheme * (loss_scheme1)

                # low-k anchor
                l_lowk = low_k_mse(y_hat, u_si1, frac=0.45)

                # physics mix (scale identical to utilities)
                loss_phys = 1e-3 * ( l_mid_norm + w_lowk * l_lowk)


                # SH free-energy hinge
                loss_energy = 0.3 * energy_penalty_sh(u_in_last, y_hat, config.DX, config.EPSILON_PARAM)

            elif USE_AC:
                # gentle ramps (same spirit as SH/PFC/MBE)
                epoch_frac = ep / max(1, (config.EPOCHS - 1))
                w_scheme = 0.32 - 0.12 * epoch_frac
                w_lowk = 0.25 + 0.60 * (epoch_frac ** 2)

                # residuals
                #l_fft = scheme_residual_fourier(u_in_last, y_hat)  # AC path uses its (explicit) residual

                # Gauss–Lobatto L2 collocation (identical form to others)
                tau_off = 1.0 / (2.0 * math.sqrt(5.0))
                l_tau1 = physics_collocation_tau_L2_AC(u_in_last, y_hat, tau=(0.5 - tau_off))
                l_tau2 = physics_collocation_tau_L2_AC(u_in_last, y_hat, tau=(0.5 + tau_off))
                l_mid_norm = 0.5 * (l_tau1 + l_tau2)

                # teacher consistency (AC semi-implicit); PGU on second step
                u_si1 = semi_implicit_step(u_in_last, config.DT, config.DX, config.EPS2)
                loss_scheme1 = F.mse_loss(y_hat, u_si1)

                with torch.no_grad():
                    u_si2 = semi_implicit_step(u_si1, config.DT, config.DX, config.EPS2)

                x2 = torch.cat([x[..., 1:], y_hat], dim=-1)
                y_hat2 = model(x2)
                y_hat2 = physics_guided_update_ac_optimal(x2[..., -1:], y_hat2, alpha_cap=0.6, low_k_snap_frac=0.45)
                loss_scheme2 = F.mse_loss(y_hat2, u_si2)
                loss_scheme = w_scheme * (0.6 * loss_scheme1 )

                # low-k anchor (stabilize coarse scales)
                l_lowk = low_k_mse(y_hat, u_si1, frac=0.45)

                # physics mix and energy (mirror SH/PFC style)
                loss_phys = 1e-3 * (0.7 * l_mid_norm + w_lowk * 0.40 * l_lowk)
                loss_energy = 0.03 * energy_penalty(u_in_last, y_hat, config.DX, config.EPS2)
            # correct
            elif USE_CH:
                # gentle ramps like SH/PFC/MBE
                epoch_frac = ep / max(1, (config.EPOCHS - 1))
                w_scheme = 0.32 - 0.12 * epoch_frac
                w_lowk = 0.25 + 0.70 * (epoch_frac ** 2)

                # --- forward (CH: hard mass projection to enforce invariance) ---
                y_pred = model(x)
                y_hat = y_pred # mass_project_pred(y_pred, u_in_last)

                # --- Fourier preconditioned semi-implicit residual (CH-aware) ---
                #l_fft = scheme_residual_fourier(u_in_last, y_hat)

                # --- L2 Gauss–Lobatto collocation (same nodes as SH/PFC/MBE) ---
                tau_off = 1.0 / (2.0 * math.sqrt(5.0))
                l_tau1 = physics_collocation_tau_L2_CH(u_in_last, y_hat, tau=(0.5 - tau_off))
                l_tau2 = physics_collocation_tau_L2_CH(u_in_last, y_hat, tau=(0.5 + tau_off))
                l_mid_norm = 0.5 * (l_tau1 + l_tau2)

                # --- teacher consistency (CH semi-implicit), with PGU on step-2 ---
                u_si1 = semi_implicit_step_ch(u_in_last, config.DT, config.DX, config.EPSILON_PARAM)
                loss_scheme1 = F.mse_loss(y_hat, u_si1)

                # with torch.no_grad():
                #    u_si2 = semi_implicit_step_ch(u_si1, config.DT, config.DX, config.EPSILON_PARAM)

                # x2 = torch.cat([x[..., 1:], y_hat], dim=-1)
                # y_hat2 = model(x2)
                # y_hat2 = physics_guided_update_ch_optimal(
                #    x2[..., -1:], y_hat2, alpha_cap=0.6, low_k_snap_frac=0.45
                # )
                # loss_scheme2 = F.mse_loss(y_hat2, u_si2)
                # loss_scheme = w_scheme * (0.6 * loss_scheme1 + 0.4 * loss_scheme2)
                loss_scheme = w_scheme * (0.6 * loss_scheme1)

                # --- spectral low-k anchor (a bit stronger for CH) ---
                l_lowk = low_k_mse(y_hat, u_si1, frac=0.50)

                # --- physics mix (same base scale), add tiny H^{-1} terms ---
                # loss_phys = 8e-3 * (l_fft + 0.6 * l_mid_norm + w_lowk * 0.70 * l_lowk)
                loss_phys = 1e-3 * ( l_mid_norm + w_lowk *  l_lowk)
                # --- energy hinge (AC/CH) + very soft mass regularizer ---
                loss_energy = energy_penalty(u_in_last, y_hat, config.DX, config.EPS2)

            elif USE_PFC:

                # project to preserve mean
                #y_hat = mass_project_pred(y_pred, u_in_last)
                y_hat = y_pred

                # --- PFC physics bundle (matches utilities) ---
                epoch_frac = ep / max(1, (config.EPOCHS - 1))
                w_scheme = 0.32 - 0.12 * epoch_frac
                w_lowk = 0.25 + 0.60 * (epoch_frac ** 2)

                #l_fft = scheme_residual_fourier(u_in_last, y_hat)
                tau_off = 1.0 / (2.0 * math.sqrt(5.0))
                l_tau1 = physics_collocation_tau_L2_PFC(u_in_last, y_hat, tau=(0.5 - tau_off))
                l_tau2 = physics_collocation_tau_L2_PFC(u_in_last, y_hat, tau=(0.5 + tau_off))
                l_mid_norm = 0.5 * (l_tau1 + l_tau2)

                # teacher + gentle second step
                u_si1 = semi_implicit_step_pfc(u_in_last, config.DT, config.DX, config.EPSILON_PARAM)
                loss_scheme1 = F.mse_loss(y_hat, u_si1)
                #with torch.no_grad():
                #    u_si2 = semi_implicit_step_pfc(u_si1, config.DT, config.DX, config.EPSILON_PARAM)
                #x2 = torch.cat([x[..., 1:], y_hat], dim=-1)
                #y_hat2 = model(x2)
                #loss_scheme2 = F.mse_loss(y_hat2, u_si2)
                loss_scheme = w_scheme * (loss_scheme1 )

                # low-k anchor
                l_lowk = low_k_mse(y_hat, u_si1, frac=0.45)

                # physics mix
                loss_phys = 1e-3 * ( l_mid_norm + w_lowk * l_lowk)

                # energy hinge
                loss_energy = 0.3 * energy_penalty_pfc(u_in_last, y_hat, config.DX, config.EPSILON_PARAM)

            elif USE_MBE:
                # project to preserve mean
                #y_hat = mass_project_pred(y_pred, u_in_last)

                epoch_frac = ep / max(1, (config.EPOCHS - 1))
                w_scheme = 0.32 - 0.12 * epoch_frac
                w_lowk = 0.25 + 0.60 * (epoch_frac ** 2)

                # residuals
                #l_fft = scheme_residual_fourier(u_in_last, y_hat)
                tau_off = 1.0 / (2.0 * math.sqrt(5.0))
                l_tau1 = physics_collocation_tau_L2_MBE(u_in_last, y_hat, tau=(0.5 - tau_off))
                l_tau2 = physics_collocation_tau_L2_MBE(u_in_last, y_hat, tau=(0.5 + tau_off))
                # l_tau_mid = physics_collocation_tau_L2_MBE(u_in_last, y_hat, tau=0.5)
                # l_mid_norm = 0.25 * (l_tau1 + l_tau2) + 0.50 * l_tau_mid
                l_mid_norm = 0.5 * (l_tau1 + l_tau2)
                # teacher consistency (+ one more gentle step)
                u_si1 = semi_implicit_step_mbe(u_in_last, config.DT, config.DX, config.EPSILON_PARAM)
                loss_scheme1 = F.mse_loss(y_hat, u_si1)
                #with torch.no_grad():
                #    u_si2 = semi_implicit_step_mbe(u_si1, config.DT, config.DX, config.EPSILON_PARAM)
                #x2 = torch.cat([x[..., 1:], y_hat], dim=-1)
                #y_hat2 = model(x2)
                # y_hat2 = physics_guided_update_mbe_optimal(y_hat, y_hat2, alpha_cap=0.6, low_k_snap_frac=0.45)
                #loss_scheme2 = F.mse_loss(y_hat2, u_si2)
                #loss_scheme = w_scheme * (0.6 * loss_scheme1 + 0.4 * loss_scheme2)
                loss_scheme = w_scheme * ( loss_scheme1 )
                # spectral low-k anchor
                l_lowk = low_k_mse(y_hat, u_si1, frac=0.50)

                # physics mix (scale identical to utilities)
                #loss_phys = 6e-3 * (1.0 * l_fft + l_mid_norm + w_lowk * 0.40 * l_lowk)
                loss_phys = 1e-3 * ( l_mid_norm + w_lowk * l_lowk)
                # energy hinge + tiny mass regularizer
                loss_energy = 0.3 * energy_penalty_mbe(u_in_last, y_hat, config.DX, config.EPSILON_PARAM)
            else:
                raise RuntimeError(f"Unknown/unsupported PROBLEM: {config.PROBLEM}")

            # ---- total loss (unchanged structure) ----
            loss_total = (1.0 - pde_weight) * loss_data + pde_weight * (loss_phys + loss_scheme + loss_energy)
            #loss_total = (1.0 - pde_weight) * loss_data + pde_weight * loss_phys
            # combined objective (identical form across modes)
            #loss_total = (1.0 - lam) * loss_data +  lam * loss_phys


            # backward
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()


            # accumulators
            data_loss_acc  += loss_data.item()
            phys_loss_acc  += loss_phys.item()
            #data_loss_acc += contrib_data.item()
            #phys_loss_acc += contrib_phys.item()
            l_mid_norm_ch_cc += l_mid_norm.item()
            energy_loss_acc += loss_energy.item()
            scheme_loss_acc += loss_scheme.item()
            #scheme_loss_acc += 0
            total_loss_acc += loss_total.item()
            n_batches      += 1
            # LR scheduler should step **per update** (see main.py change)
            scheduler.step()

        # eval
        model.eval()
        with torch.no_grad():
            rels = []
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_pred = model(x)
                rels.append(relative_l2(y_pred, y))
            test_rel = torch.cat(rels, dim=0).mean().item()

        #scheduler.step()
        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']
        print(f"{ep:5d} | {t2-t1:7.3f} | "
              f"{data_loss_acc/n_batches:8.3e} | {phys_loss_acc/n_batches:8.3e} | {total_loss_acc/n_batches:9.3e} | "
              f"{test_rel:10.3e} |  {energy_loss_acc:10.3e} |  {scheme_loss_acc:10.3e} | {l_mid_norm_ch_cc:10.3e} | {lr: .2e}")

# ---------------------
# Evaluation: rollout
# ---------------------
def rollout_autoregressive(model, traj_np, T_in, Nt=100):
    """
    traj_np: (Nt+1, S,S,S) ground truth trajectory for one sample
    returns pred: same shape, with pred[0:T_in]=gt[0:T_in], rest autoregressive
    """
    device = next(model.parameters()).device
    Nt_plus1, Sx, Sy, Sz = traj_np.shape
    assert Nt_plus1 >= Nt + 1
    # put full GT trajectory on device as (Nt+1, Sx,Sy,Sz)
    traj_torch = torch.from_numpy(traj_np).to(device)  # (Nt+1,Sx,Sy,Sz)
    pred = traj_torch.clone()                          # will be overwritten by predictions
    model.eval()
    for t in range(T_in - 1, Nt):
        x_win = pred[t - (T_in - 1):t + 1]
        # reorder to (1, Sx, Sy, Sz, T_in)  -- SAME as training
        x = x_win.permute(1, 2, 3, 0).unsqueeze(0)     # (1,Sx,Sy,Sz,T_in)
        with torch.no_grad():
            y_next = model(x)  # (1,Sx,Sy,Sz,1) typically
        # y_next: (1,Sx,Sy,Sz,1) → (Sx,Sy,Sz)
        y_step = y_next.squeeze(0).squeeze(-1)
        # store back on GPU in pred
        pred[t + 1] = y_step
    # only convert ONCE at the end
    pred_np = pred.detach().cpu().numpy()  # (Nt+1,Sx,Sy,Sz)
    return pred_np


def relative_l2_scalar(a, b, eps=1e-12):
    num = np.linalg.norm(a.ravel() - b.ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return num / den

def evaluate_stats_and_plot(model, mat_path, test_ids, times):
    import matplotlib
    matplotlib.use('TkAgg')
    import h5py, numpy as np
    import matplotlib.pyplot as plt

    def sym_vlims(A, sym_frac=0.995):
        m = np.mean(A)
        a = np.quantile(np.abs(A - m), sym_frac)
        return m - a, m + a


    with h5py.File(mat_path, "r") as f:
        dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
        Nz, Ny, Nx, Nt, Ns = dset.shape
        assert Nt == config.SAVED_STEPS

        rel_errors = {t: [] for t in times}

        # pick first test id for plotting
        #pid = int(test_ids[0])
        mode = getattr(config, "TEST_MODE", "random")
        if mode == "manual":
            pick = int(getattr(config, "TEST_PICK", 0))
            pid = int(test_ids[pick % len(test_ids)])  # pick a specific ID from the already-random test_ids
        else:
            pid = int(test_ids[0])
        print(f"[Eval] Visualization sample id (from test_ids): {pid}")

        gt_raw = np.array(dset[:, :, :, :, pid], dtype=np.float32)  # (Nz,Ny,Nx,Nt)
        gt = np.transpose(gt_raw, (3,2,1,0))                        # (Nt,Nx,Ny,Nz)
        # ---- Inference timing: one full rollout for a new PDE instance ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make sure GPU is idle before timing
        t_inf_start = default_timer()

        pred = rollout_autoregressive(model, gt, config.T_IN_CHANNELS,
                                      Nt=config.TOTAL_TIME_STEPS)

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure all kernels are finished
        t_inf_end = default_timer()

        inf_time = t_inf_end - t_inf_start
        print(f"[Timing] Inference time for one rollout (Nt={config.TOTAL_TIME_STEPS}) "
              f"= {inf_time:.4f} s ({inf_time / 60:.4f} min)")


        # stats across test ids
        for sid in test_ids:
            gt_raw = np.array(dset[:, :, :, :, sid], dtype=np.float32)
            gt_s = np.transpose(gt_raw, (3,2,1,0))
            pred_s = rollout_autoregressive(model, gt_s, config.T_IN_CHANNELS,
                                            Nt=config.TOTAL_TIME_STEPS)
            for t in times:
                num = np.linalg.norm(pred_s[t].ravel() - gt_s[t].ravel())
                den = np.linalg.norm(gt_s[t].ravel()) + 1e-12
                rel_errors[t].append(num/den)

        # print stats
        all_vals = []
        print("\nRelative L2 error stats:")
        for t in times:
            arr = np.array(rel_errors[t])
            print(f"t={t:3d}: mean={arr.mean():.4e}  min={arr.min():.4e}  max={arr.max():.4e}  std={arr.std():.4e}")
            all_vals.extend(arr.tolist())
        all_vals = np.array(all_vals)
        print(f"OVERALL frames {times}: mean={all_vals.mean():.4e}  min={all_vals.min():.4e}  "
              f"max={all_vals.max():.4e}  std={all_vals.std():.4e}")

        # 3×len(times) subplot (central z-slice)
        S = gt.shape[1]; zc = S // 2
        fig, axes = plt.subplots(3, len(times), figsize=(4*len(times), 9))
        for j, t in enumerate(times):
            exact = gt[t, :, :, zc]
            predt = pred[t, :, :, zc]
            rel   = np.abs(predt - exact) / (np.abs(exact) + 1e-8)

            v0, V0 = sym_vlims(exact)
            v1, V1 = sym_vlims(predt)
            im0 = axes[0, j].imshow(exact, origin='lower', cmap='RdBu_r', vmin=v0, vmax=V0)
            im1 = axes[1, j].imshow(predt, origin='lower', cmap='RdBu_r', vmin=v1, vmax=V1)

            #im0 = axes[0, j].imshow(exact, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, j].set_title(f"Exact t={t}");  fig.colorbar(im0, ax=axes[0, j], shrink=0.8)
            #im1 = axes[1, j].imshow(predt, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, j].set_title(f"Pred t={t}");   fig.colorbar(im1, ax=axes[1, j], shrink=0.8)
            im2 = axes[2, j].imshow(rel, origin='lower', cmap='viridis')
            axes[2, j].set_title(f"Rel. L2 (px) t={t}"); fig.colorbar(im2, ax=axes[2, j], shrink=0.8)

            for r in range(3):
                axes[r, j].set_xlabel('x'); axes[r, j].set_ylabel('y')
        plt.tight_layout(); plt.show()