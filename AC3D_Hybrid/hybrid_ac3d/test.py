import h5py, numpy as np, torch, random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import config
from itertools import cycle

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


################

if USE_AC:
    # Gauss–Lobatto L2 collocation (identical form to others)
    tau_off = 1.0 / (2.0 * math.sqrt(5.0))
    l_tau1 = physics_collocation_tau_L2_AC(u_in_last, y_hat, tau=(0.5 - tau_off))
    l_tau2 = physics_collocation_tau_L2_AC(u_in_last, y_hat, tau=(0.5 + tau_off))
    l_mid_norm = 0.5 * (l_tau1 + l_tau2)

    # teacher consistency (AC semi-implicit); PGU on second step
    u_si1 = semi_implicit_step(u_in_last, config.DT, config.DX, config.EPS2)
    # low-k anchor (stabilize coarse scales)
    l_lowk = low_k_mse(y_hat, u_si1, frac=0.45)

    # physics mix and energy (mirror SH/PFC style)
    loss_phys = l_mid_norm + l_lowk
    loss_energy = energy_penalty(u_in_last, y_hat, config.DX, config.EPS2)



#################

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



###############



# === ADD THIS IN utilities.py ===

def train_ac_beamstyle(
    model, train_loader, test_loader, optimizer, device,
    lambda_tradeoff=None, data_loss_scaling_factor=1.0,
    use_lbfgs=True, lbfgs_max_iter=20
):
    """
    Beam-style hybrid for AC ONLY:
    J = (1-λ) * α * L_data + λ * L_phys,
    with L_phys = GL3 collocation + 0.5 * energy hinge (identity).
    No EMA, no outer scales, no curriculum. Exact beam recipe.

    Args:
        lambda_tradeoff: float in [0,1]. (0=data-only, 1=physics-only)
        data_loss_scaling_factor: α
        use_lbfgs: if True, do a short L-BFGS polish after Adam
    """

    assert config.PROBLEM == 'AC3D', "train_ac_beamstyle is AC-only."
    lam = config.PDE_WEIGHT if (lambda_tradeoff is None) else float(lambda_tradeoff)
    alpha = float(data_loss_scaling_factor)

    # simple cosine LR decay per-iteration (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.EPOCHS*getattr(config, "STEPS_PER_EPOCH", 20))
    )

    steps_per_epoch = getattr(config, "STEPS_PER_EPOCH_EFF", getattr(config, "STEPS_PER_EPOCH", 20))
    train_iter = cycle(train_loader)

    print("Phase 1: Adam optimization (beam-style combine)")
    print("Epoch |   Time   | Loss Hyb. | loss_phys. | Loss data. | Test relL2 |  LR")

    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        n_batches = 0
        hyb_acc = phys_acc = data_acc = 0.0

        for _ in range(steps_per_epoch):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)

            optimizer.zero_grad(set_to_none=True)

            # forward
            y_pred = model(x)

            # --- data loss (unweighted) ---
            loss_data_unweighted = F.mse_loss(y_pred, y)

            # --- AC physics (raw, unscaled) ---
            # GL3 collocation + 0.5 * energy hinge (your functions)
            l_gl3 = physics_collocation_GL3_AC(u_in_last, y_pred, normalize=True)
            l_energy = energy_dissipation_identity_AC(u_in_last, y_pred)
            loss_phys_raw = l_gl3 + 0.5 * l_energy

            # --- beam-style combine ---
            if lam == 0.0:
                loss_total = loss_data_unweighted
            elif lam == 1.0:
                loss_total = loss_phys_raw
            else:
                loss_total = (1.0 - lam) * (alpha * loss_data_unweighted) + lam * loss_phys_raw

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            hyb_acc  += loss_total.item()
            phys_acc += loss_phys_raw.item()
            data_acc += loss_data_unweighted.item()
            n_batches += 1

        # eval
        model.eval()
        with torch.no_grad():
            rels = []
            for x_t, y_t in test_loader:
                x_t = x_t.to(device, non_blocking=True)
                y_t = y_t.to(device, non_blocking=True)
                y_hat = model(x_t)
                rels.append(relative_l2(y_hat, y_t))
            test_rel = torch.cat(rels, dim=0).mean().item()

        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']
        print(f"{ep:5d} | {t2-t1:7.3f} | "
              f"{hyb_acc/n_batches:9.4e} | {phys_acc/n_batches:9.4e} | {data_acc/n_batches:10.4e} | "
              f"{test_rel:10.3e} | {lr: .2e}")

    # --- Phase 2: optional L-BFGS polish (like the beam script) ---
    '''
    if use_lbfgs:
        print("\nPhase 2: L-BFGS optimization (short polish)")
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter,
                                  history_size=100, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            # take one representative batch
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]
            y_pred = model(x)
            loss_data_unweighted = F.mse_loss(y_pred, y)
            l_gl3 = physics_collocation_GL3_AC(u_in_last, y_pred, normalize=True)
            l_energy = energy_dissipation_identity_AC(u_in_last, y_pred)
            loss_phys_raw = l_gl3 + 0.5 * l_energy

            if lam == 0.0:
                loss_total = loss_data_unweighted
            elif lam == 1.0:
                loss_total = loss_phys_raw
            else:
                loss_total = (1.0 - lam) * (alpha * loss_data_unweighted) + lam * loss_phys_raw

            loss_total.backward()
            return loss_total

        for _ in range(30):
            lbfgs.step(closure)
    '''


def train_ch_beamstyle(
    model, train_loader, test_loader, optimizer, device,
    lambda_tradeoff=None, data_loss_scaling_factor=1.0,
    use_lbfgs=True, lbfgs_max_iter=20
):
    """
    Beam-style hybrid for CH ONLY:
    J = (1-λ) * α * L_data + λ * L_phys,
    with L_phys = GL3 collocation + 0.5 * energy hinge (identity).
    No EMA, no outer scales, no curriculum. Exact beam recipe.

    Args:
        lambda_tradeoff: float in [0,1]. (0=data-only, 1=physics-only)
        data_loss_scaling_factor: α
        use_lbfgs: if True, do a short L-BFGS polish after Adam
    """

    assert config.PROBLEM == 'CH3D', "train_ch_beamstyle is CH-only."
    lam = config.PDE_WEIGHT if (lambda_tradeoff is None) else float(lambda_tradeoff)
    alpha = float(data_loss_scaling_factor)

    # simple cosine LR decay per-iteration (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.EPOCHS*getattr(config, "STEPS_PER_EPOCH", 20))
    )

    steps_per_epoch = getattr(config, "STEPS_PER_EPOCH_EFF", getattr(config, "STEPS_PER_EPOCH", 20))
    train_iter = cycle(train_loader)
    # -------- EMA normalization state --------
    EMA_BETA = 0.98
    ema_data = None  # running mean of raw data loss
    ema_phys = None  # running mean of raw physics loss

    print("Phase 1: Adam optimization (beam-style combine)")
    print("Epoch |   Time   | Loss Hyb. | loss_phys. | Loss data. | Test relL2 |  LR")

    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        n_batches = 0
        hyb_acc = phys_acc = data_acc = 0.0

        for _ in range(steps_per_epoch):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)

            optimizer.zero_grad(set_to_none=True)

            # forward
            y_pred = model(x)

            # CH mass projection (one line, preserves beam-style)
            mean_in = u_in_last.mean(dim=(1, 2, 3, 4), keepdim=True)
            mean_out = y_pred.mean(dim=(1, 2, 3, 4), keepdim=True)
            y_pred = y_pred - mean_out + mean_in

            # --- data loss (unweighted) ---
            loss_data_unweighted = F.mse_loss(y_pred, y)

            # --- CH physics (raw, unscaled) ---
            #l_gl3 = physics_collocation_GL3_CH(u_in_last, y_pred, normalize=True)
            ##l_gl3 = physics_collocation_GL3_CH_Hm1(u_in_last, y_pred, normalize=True)
            ##l_energy = energy_dissipation_identity_CH(u_in_last, y_pred)
            ##loss_phys_raw = l_gl3 + 0.5 * l_energy

            ##
            # --- CH physics (raw, unscaled) ---
            l_gl5_hm2 = physics_collocation_GL5_CH_Hm2(u_in_last, y_pred, normalize=True)
            l_energy = energy_dissipation_identity_CH(u_in_last, y_pred)

            # (optional but recommended, tiny) teacher anchor to suppress very-long drifts
            u_si1 = semi_implicit_step_ch(u_in_last, config.DT, config.DX, config.EPSILON_PARAM)
            l_lowk = low_k_mse(y_pred, u_si1, frac=0.50)
            # NEW: two-step semi-implicit consistency (tiny)
            #l_scheme2 = ch_two_step_consistency(x, model, alpha_cap=0.4)

            # keep weights small; these worked well in practice for CH
            loss_phys_raw = l_gl5_hm2 + 1.0 * l_energy + 0.5 * l_lowk

            # --- initialize EMAs on first batch ---
            if ema_data is None:
                ema_data = float(loss_data_unweighted.detach())
                ema_phys = float(loss_phys_raw.detach())

            # --- update EMAs (detach to avoid gradient flow) ---
            ema_data = EMA_BETA * ema_data + (1.0 - EMA_BETA) * float(loss_data_unweighted.detach())
            ema_phys = EMA_BETA * ema_phys + (1.0 - EMA_BETA) * float(loss_phys_raw.detach())

            # --- normalized, dimensionless losses ---
            loss_data_norm = loss_data_unweighted / (ema_data + 1e-8)
            loss_phys_norm = loss_phys_raw / (ema_phys + 1e-8)

            # --- combine (always use normalized terms, even at lam=0/1) ---
            loss_total =  (1.0 - lam) * ( loss_data_norm) + lam * loss_phys_norm



            '''
            # --- beam-style combine ---
            if lam == 0.0:
                loss_total = loss_data_unweighted
            elif lam == 1.0:
                loss_total = loss_phys_raw
            else:
                loss_total = (1.0 - lam) * (alpha * loss_data_unweighted) + lam * loss_phys_raw
            '''

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            hyb_acc  += loss_total.item()
            phys_acc += loss_phys_raw.item()
            data_acc += loss_data_unweighted.item()
            n_batches += 1

        # eval
        model.eval()
        with torch.no_grad():
            rels = []
            for x_t, y_t in test_loader:
                x_t = x_t.to(device, non_blocking=True)
                y_t = y_t.to(device, non_blocking=True)
                y_hat = model(x_t)
                mean_in = x_t[..., -1:].mean(dim=(1, 2, 3, 4), keepdim=True)
                mean_out = y_hat.mean(dim=(1, 2, 3, 4), keepdim=True)
                y_hat = y_hat - mean_out + mean_in

                rels.append(relative_l2(y_hat, y_t))
            test_rel = torch.cat(rels, dim=0).mean().item()

        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']
        print(f"{ep:5d} | {t2-t1:7.3f} | "
              f"{hyb_acc/n_batches:9.4e} | {phys_acc/n_batches:9.4e} | {data_acc/n_batches:10.4e} | "
              f"{test_rel:10.3e} | {lr: .2e}")

    # --- Phase 2: optional L-BFGS polish (like the beam script) ---

    if use_lbfgs:
        print("\nPhase 2: L-BFGS optimization (short polish)")
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter,
                                  history_size=100, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            # take one representative batch
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]
            y_pred = model(x)
            loss_data_unweighted = F.mse_loss(y_pred, y)
            l_gl3 = physics_collocation_GL3_CH(u_in_last, y_pred, normalize=True)
            l_energy = energy_dissipation_identity_CH(u_in_last, y_pred)
            loss_phys_raw = l_gl3 + 0.5 * l_energy

            if lam == 0.0:
                loss_total = loss_data_unweighted
            elif lam == 1.0:
                loss_total = loss_phys_raw
            else:
                loss_total = (1.0 - lam) * (alpha * loss_data_unweighted) + lam * loss_phys_raw

            loss_total.backward()
            return loss_total

        for _ in range(30):
            lbfgs.step(closure)



def train_sh_beamstyle(
    model, train_loader, test_loader, optimizer, device,
    lambda_tradeoff=None, data_loss_scaling_factor=1.0,
    use_lbfgs=True, lbfgs_max_iter=20
):
    """
    Beam-style hybrid for SH ONLY:
    J = (1-λ) * α * L_data + λ * L_phys,
    with L_phys = GL3 collocation + 0.5 * energy hinge (SH version).
    Mirrors the AC recipe (no mass projection, no teacher anchor).
    """
    assert config.PROBLEM == 'SH3D', "train_sh_beamstyle is SH-only."
    lam = config.PDE_WEIGHT if (lambda_tradeoff is None) else float(lambda_tradeoff)
    alpha = float(data_loss_scaling_factor)

    # cosine per-iteration LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.EPOCHS * getattr(config, "STEPS_PER_EPOCH", 20))
    )

    steps_per_epoch = getattr(config, "STEPS_PER_EPOCH_EFF",
                              getattr(config, "STEPS_PER_EPOCH", 20))
    train_iter = cycle(train_loader)

    print("Phase 1: Adam optimization (beam-style combine)")
    print("Epoch |   Time   | Loss Hyb. | loss_phys. | Loss data. | Test relL2 |  LR")

    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        n_batches = 0
        hyb_acc = phys_acc = data_acc = 0.0

        for _ in range(steps_per_epoch):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)

            optimizer.zero_grad(set_to_none=True)

            # forward
            y_pred = model(x)

            # --- data loss (unweighted) ---
            loss_data_unweighted = F.mse_loss(y_pred, y)

            # --- SH physics (raw, unscaled) ---
            l_gl5_hm2 = physics_collocation_GL5_SH_Hm2(u_in_last, y_pred, normalize=True)
            l_energy = energy_dissipation_identity_SH(u_in_last, y_pred)

            # tiny teacher: one semi-implicit step like MATLAB
            u_si1 = semi_implicit_step_sh(u_in_last, config.DT, config.DX, config.EPSILON_PARAM)
            l_lowk = low_k_mse(y_pred, u_si1, frac=0.50)  # reuse your existing helper

            # weights: start conservative; you can nudge 1.0→0.5/2.0 later if needed
            loss_phys_raw = l_gl5_hm2 + 0.5 * l_energy + 0.1 * l_lowk

            # --- beam-style combine ---
            if lam == 0.0:
                loss_total = loss_data_unweighted
            elif lam == 1.0:
                loss_total = loss_phys_raw
            else:
                loss_total = (1.0 - lam) * (alpha * loss_data_unweighted) + lam * loss_phys_raw

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            hyb_acc  += loss_total.item()
            phys_acc += loss_phys_raw.item()
            data_acc += loss_data_unweighted.item()
            n_batches += 1

        # eval
        model.eval()
        with torch.no_grad():
            rels = []
            for x_t, y_t in test_loader:
                x_t = x_t.to(device, non_blocking=True)
                y_t = y_t.to(device, non_blocking=True)
                y_hat = model(x_t)
                rels.append(relative_l2(y_hat, y_t))
            test_rel = torch.cat(rels, dim=0).mean().item()

        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']
        if ep % 10 == 0:
            print(f"{ep:5d} | {t2-t1:7.3f} | "
                  f"{hyb_acc/n_batches:9.4e} | {phys_acc/n_batches:9.4e} | {data_acc/n_batches:10.4e} | "
                  f"{test_rel:10.3e} | {lr: .2e}")

    # --- optional short L-BFGS polish (kept off by default, identical to AC scaffold) ---




##################

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

    w_mid, w_fft = 1.0, 1.0
    CLIP_NORM = 1.0
    w_mm = 0.0 if (USE_CH or USE_SH) else 1.0  # CH & SH: disable MM; AC: unchanged



    print("Epoch |   Time   | DataLoss | PhysLoss | TotalLoss | Test relL2 | energy_loss | scheme_loss | l_mid_norm_ch_cc | LR")

    # FIXED number of updates per epoch (independent of N_TRAIN)
    #steps_per_epoch = getattr(config, "STEPS_PER_EPOCH", 20)  # or any value you want
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
            if pde_weight == 0:
                loss_data =  F.mse_loss(y_pred, y)
            else:
                loss_data = config.DATA_LOSS_SCALE * F.mse_loss(y_pred, y)

            if USE_AC:

                # Gauss–Lobatto L2 collocation (identical form to others)
                tau_off = 1.0 / (2.0 * math.sqrt(5.0))
                l_tau1 = physics_collocation_tau_L2_AC(u_in_last, y_hat, tau=(0.5 - tau_off))
                l_tau2 = physics_collocation_tau_L2_AC(u_in_last, y_hat, tau=(0.5 + tau_off))
                l_mid_norm = 0.5 * (l_tau1 + l_tau2)

                # teacher consistency (AC semi-implicit); PGU on second step
                u_si1 = semi_implicit_step(u_in_last, config.DT, config.DX, config.EPS2)
                # low-k anchor (stabilize coarse scales)
                l_lowk = low_k_mse(y_hat, u_si1, frac=0.45)

                # physics mix and energy (mirror SH/PFC style)
                loss_phys =   l_mid_norm +  l_lowk
                loss_energy = energy_penalty(u_in_last, y_hat, config.DX, config.EPS2)



            loss_total = (1.0 - pde_weight) * loss_data + pde_weight * (loss_phys  + loss_energy)



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
            #scheme_loss_acc += loss_scheme.item()
            scheme_loss_acc += 0
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
    pred = np.zeros_like(traj_np, dtype=np.float32)
    pred[:T_in] = traj_np[:T_in]
    for t in range(T_in-1, Nt):
        x = torch.from_numpy(pred[t-(T_in-1):t+1]).permute(1,2,3,0).unsqueeze(0).to(device)  # (1,S,S,S,T_in)
        with torch.no_grad():
            y_next = model(x)

            if config.PROBLEM == 'SH3D':
                # mild correction toward the SH semi-implicit teacher in Fourier
                #y_next = physics_guided_update_sh_optimal(x[..., -1:], y_next, alpha_cap=0.6)
                #y_next = model(x)
                y_next = physics_guided_update_sh_optimal(
                    x[..., -1:], model(x), alpha_cap=0.6, low_k_snap_frac=0.45
                )
            elif config.PROBLEM == 'PFC3D':
                y_next = physics_guided_update_pfc_optimal(
                    x[..., -1:], model(x), alpha_cap=0.6, low_k_snap_frac=0.45
                )

            elif config.PROBLEM == 'MBE3D':
                # keep it simple (no extra PGU function defined for MBE): use raw model step
                #y_next = y_next
                y_next = physics_guided_update_mbe_optimal(x[..., -1:], model(x), alpha_cap=0.6, low_k_snap_frac=0.45)

            elif config.PROBLEM == 'CH3D':
                #y_next = physics_guided_update_ch_optimal(
                #    x[..., -1:], model(x), alpha_cap=0.6, low_k_snap_frac=0.45
                #)
                #y_next = y_next
                y_next = model(x)
                m_in = x[..., -1:].mean(dim=(1, 2, 3, 4), keepdim=True)
                m_out = y_next.mean(dim=(1, 2, 3, 4), keepdim=True)
                y_next = y_next - m_out + m_in

                y_next = physics_guided_update_ch_optimal(
                    x[..., -1:], y_next, alpha_cap=0.2, low_k_snap_frac=0.45
                )

            elif config.PROBLEM == 'AC3D':
                y_next = physics_guided_update_ac_optimal(
                    x[..., -1:], model(x), alpha_cap=0.3, low_k_snap_frac=0.45
                )




            y_np = y_next.squeeze(0).squeeze(-1).detach().cpu().numpy()

        pred[t + 1] = y_np
    return pred

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
        pred = rollout_autoregressive(model, gt, config.T_IN_CHANNELS,
                                      Nt=config.TOTAL_TIME_STEPS)

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



#################
################# New modified AC sweep file based on 3 point Gouss Legander

# main_sweep_ac3d.py
# AC3D sweep with identical budgeting/logging policy as CH/SH/MBE/PFC sweeps.
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
from itertools import cycle
from timeit import default_timer

import config as CFG
import config as config  # keep an alias so the exact training loop uses `config.*`
from networks import FNO4d, TNO3d
from utilities import (
    build_loaders,
    relative_l2,
    rollout_autoregressive,
)
from functions import (
    # --- beam-style AC physics used below ---
    physics_collocation_GL3_AC,
    energy_dissipation_identity_AC,
)

# --------------------
# Sweep parameters
# --------------------
DEFAULT_PDE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_MODELS      = ['FNO4d', 'TNO3d']   # TNO3d labeled "MHNO" in saved results
DEFAULT_N_TRAINS    = [20, 30, 50]

# Phase frames for figures
TIME_FRAMES = [0, 20, 40, 60, 80, 100]  # indices
VOLUME_DOWNSAMPLE = 1  # set >1 to shrink 3D volumes in the mat for speed/size

# Save dirs (AC3D)
MODELS_DIR = Path("./AC3d_models")
MAT_DIR    = Path("./AC3d_mat")
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
# TRAIN LOOP: exact beam-style AC trainer (verbatim) + thin wrapper for logs
# --------------------
def train_ac_beamstyle(
    model, train_loader, test_loader, optimizer, device,
    lambda_tradeoff=None, data_loss_scaling_factor=1.0,
    use_lbfgs=True, lbfgs_max_iter=20
):
    """
    Beam-style hybrid for AC ONLY:
    J = (1-λ) * α * L_data + λ * L_phys,
    with L_phys = GL3 collocation + 0.5 * energy hinge (identity).
    No EMA, no outer scales, no curriculum. Exact beam recipe.

    Args:
        lambda_tradeoff: float in [0,1]. (0=data-only, 1=physics-only)
        data_loss_scaling_factor: α
        use_lbfgs: if True, do a short L-BFGS polish after Adam
    """

    assert config.PROBLEM == 'AC3D', "train_ac_beamstyle is AC-only."
    lam = config.PDE_WEIGHT if (lambda_tradeoff is None) else float(lambda_tradeoff)
    alpha = float(data_loss_scaling_factor)

    # simple cosine LR decay per-iteration (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.EPOCHS*getattr(config, "STEPS_PER_EPOCH", 20))
    )

    steps_per_epoch = getattr(config, "STEPS_PER_EPOCH_EFF", getattr(config, "STEPS_PER_EPOCH", 20))
    train_iter = cycle(train_loader)

    print("Phase 1: Adam optimization (beam-style combine)")
    print("Epoch |   Time   | Loss Hyb. | loss_phys. | Loss data. | Test relL2 |  LR")

    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        n_batches = 0
        hyb_acc = phys_acc = data_acc = 0.0

        for _ in range(steps_per_epoch):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)

            optimizer.zero_grad(set_to_none=True)

            # forward
            y_pred = model(x)

            # --- data loss (unweighted) ---
            loss_data_unweighted = F.mse_loss(y_pred, y)

            # --- AC physics (raw, unscaled) ---
            # GL3 collocation + 0.5 * energy hinge (your functions)
            l_gl3 = physics_collocation_GL3_AC(u_in_last, y_pred, normalize=True)
            l_energy = energy_dissipation_identity_AC(u_in_last, y_pred)
            loss_phys_raw = l_gl3 + 0.5 * l_energy

            # --- beam-style combine ---
            if lam == 0.0:
                loss_total = loss_data_unweighted
            elif lam == 1.0:
                loss_total = loss_phys_raw
            else:
                loss_total = (1.0 - lam) * (alpha * loss_data_unweighted) + lam * loss_phys_raw

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            hyb_acc  += loss_total.item()
            phys_acc += loss_phys_raw.item()
            data_acc += loss_data_unweighted.item()
            n_batches += 1

        # eval
        model.eval()
        with torch.no_grad():
            rels = []
            for x_t, y_t in test_loader:
                x_t = x_t.to(device, non_blocking=True)
                y_t = y_t.to(device, non_blocking=True)
                y_hat = model(x_t)
                rels.append(relative_l2(y_hat, y_t))
            test_rel = torch.cat(rels, dim=0).mean().item()

        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']
        # light print
        if ep % 10 == 0:
            print(f"{ep:5d} | {t2-t1:7.3f} | "
                  f"{hyb_acc/n_batches:9.4e} | {phys_acc/n_batches:9.4e} | {data_acc/n_batches:10.4e} | "
                  f"{test_rel:10.3e} | {lr: .2e}")

    # --- Phase 2: optional L-BFGS polish (like the beam script) ---
    '''
    if use_lbfgs:
        print("\nPhase 2: L-BFGS optimization (short polish)")
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=lbfgs_max_iter,
                                  history_size=100, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            # take one representative batch
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]
            y_pred = model(x)
            loss_data_unweighted = F.mse_loss(y_pred, y)
            l_gl3 = physics_collocation_GL3_AC(u_in_last, y_pred, normalize=True)
            l_energy = energy_dissipation_identity_AC(u_in_last, y_pred)
            loss_phys_raw = l_gl3 + 0.5 * l_energy

            if lam == 0.0:
                loss_total = loss_data_unweighted
            elif lam == 1.0:
                loss_total = loss_phys_raw
            else:
                loss_total = (1.0 - lam) * (alpha * loss_data_unweighted) + lam * loss_phys_raw

            loss_total.backward()
            return loss_total

        for _ in range(30):
            lbfgs.step(closure)
    '''

# Thin wrapper (keeps the sweep unchanged: returns a logs dict the rest of the code expects)
def train_fno_hybrid_LOGGING_STEPBASED(model, train_loader, test_loader, optimizer, scheduler, device, pde_weight=None):
    # call the exact beam-style trainer
    lam = CFG.PDE_WEIGHT if pde_weight is None else pde_weight
    alpha = float(getattr(CFG, "DATA_LOSS_SCALE", 1.0))
    train_ac_beamstyle(
        model, train_loader, test_loader, optimizer, device,
        lambda_tradeoff=lam, data_loss_scaling_factor=alpha,
        use_lbfgs=True, lbfgs_max_iter=30
    )
    # build placeholder logs so downstream saving works without changing anything else
    logs = {k: [] for k in [
        'epoch', 'data_loss', 'phys_loss', 'energy_loss', 'scheme_loss',
        'total_loss', 'test_relL2', 'l_mid_norm_ac', 'lr'
    ]}
    for ep in range(CFG.EPOCHS):
        logs['epoch'].append(ep)
        logs['data_loss'].append(0.0)
        logs['phys_loss'].append(0.0)
        logs['energy_loss'].append(0.0)
        logs['scheme_loss'].append(0.0)
        logs['total_loss'].append(0.0)
        logs['test_relL2'].append(0.0)
        logs['l_mid_norm_ac'].append(0.0)
        logs['lr'].append(0.0)
    for k in logs:
        logs[k] = _asF32F(np.array(logs[k], dtype=np.float32))
    return logs

def set_seeds(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    assert CFG.PROBLEM == 'AC3D', f"config.PROBLEM must be 'AC3D' (got {CFG.PROBLEM})"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MAT_DIR.mkdir(parents=True, exist_ok=True)

    print("Using device:", CFG.DEVICE)
    print("TIME_FRAMES:", TIME_FRAMES)

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
                state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                torch.save({
                    'state_dict': state_dict_cpu,
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
                }, ckpt_path, _use_new_zipfile_serialization=True)

                del state_dict_cpu  # <-- put RIGHT HERE

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

    mat_name = f"AC3D_sweep_results_{int(time.time())}.mat"
    mat_path = MAT_DIR / mat_name
    #savemat(str(mat_path), out, do_compression=not FAST_SAVE)
    savemat(
        str(mat_path),
        out,
        do_compression=False,  # already fastest
        long_field_names=True,  # skip legacy 31-char checks
        format='5'  # MATLAB v5 format (same as default)
    )
    print(f"\nSaved results MAT to: {mat_path.resolve()}")
    print(f"Models saved under:   {MODELS_DIR.resolve()}")

if __name__ == "__main__":
    main()
