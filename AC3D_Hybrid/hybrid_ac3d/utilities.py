import h5py, numpy as np, torch, random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from timeit import default_timer
import config

from functions import (
    mu_ac,
    energy_penalty,
    physics_residual_midpoint,
    physics_residual_random_collocation,
    physics_residual_weak_lowk,
    physics_residual_interface_weighted,
    energy_dissipation_identity_loss_midpoint,
)

from torch.amp import GradScaler, autocast

import matplotlib
matplotlib.use('TkAgg')


# ---------------------
# Debug terms (for logs)
# ---------------------
def _debug_terms(u_in_last, y_pred):
    """Return u_t^2 mean and μ^2 mean for logging."""
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    up = y_pred.squeeze(-1)
    u0 = u_in_last.squeeze(-1)
    ut = (up - u0) / dt
    mu = mu_ac(up, dx, eps2, dealias=True)
    debug_ut = ut.pow(2).mean()
    debug_mu = config.DEBUG_MU_SCALE * mu.pow(2).mean()
    return debug_ut, debug_mu


# ---------------------
# Dataset: hold full trajectories in RAM
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
                traj = np.transpose(raw, (3, 2, 1, 0))               # (Nt,Nx,Ny,Nz)
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
# Collate: trajectories -> (x,y) windows
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
            x = x.permute(1, 2, 3, 0).contiguous()   # (S,S,S,T_in)
            y = y.unsqueeze(-1).contiguous()         # (S,S,S,1)
            xs.append(x); ys.append(y)
        x = torch.stack(xs, dim=0)         # (B,S,S,S,T_in)
        y = torch.stack(ys, dim=0)         # (B,S,S,S,1)
        if y_norm is not None:
            x = (x - y_norm.m)/y_norm.s
            y = (y - y_norm.m)/y_norm.s
        return x, y
    return _collate


# ---------------------
# Loaders
# ---------------------
def build_loaders():
    rng = np.random.default_rng(config.SEED)
    all_ids = np.arange(600)
    rng.shuffle(all_ids)

    base = AC3DTrajectoryDataset(config.MAT_DATA_PATH, all_ids)
    # split: train / test / rest (ignored)
    train_dataset, test_dataset, _ = random_split(
        base, [config.N_TRAIN, config.N_TEST, len(base)-config.N_TRAIN-config.N_TEST],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    normalizers = [base.normalizer_x, base.normalizer_y]

    collate = make_windowed_collate(
        T_in=config.T_IN_CHANNELS, t_min=0, t_max=config.TOTAL_TIME_STEPS-1,
        normalized=False, normalizers=normalizers
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True,
                             collate_fn=collate)
    # keep test ids for external evaluation if desired
    test_indices = [all_ids[i] for i in range(config.N_TRAIN, config.N_TRAIN+config.N_TEST)]
    return train_loader, test_loader, test_indices, normalizers


# ---------------------
# Metrics
# ---------------------
def relative_l2(a, b, eps=1e-12):
    diff = (a - b).flatten(start_dim=1)
    denom = b.flatten(start_dim=1)
    num = torch.sqrt(torch.sum(diff**2, dim=1) + eps)
    den = torch.sqrt(torch.sum(denom**2, dim=1) + eps)
    return (num / den)  # (B,)


def model_has_complex_params(model):
    for p in model.parameters():
        if p.is_complex():
            return True
    for b in model.buffers():
        if b.is_complex():
            return True
    return False


# ---------------------
# Hybrid trainer
# ---------------------
def train_fno_hybrid(model, train_loader, test_loader, optimizer, scheduler, device, pde_weight=None):
    """
    Train with dimensionless, balanced physics losses.
    - Physics losses are normalized by their own EMA scales.
    - PDE weight ramps up for a few epochs to avoid early instability.
    - Setting config.PDE_WEIGHT=0 yields pure data-driven training (no physics computed).
    """
    pde_weight_target = config.PDE_WEIGHT if pde_weight is None else pde_weight

    amp_allowed = config.USE_AMP and (not model_has_complex_params(model))
    scaler = GradScaler('cuda', enabled=amp_allowed)

    # Per-term running normalizers (EMA) for balance
    ema = {
        'rand': torch.tensor(1.0, device=device),
        'mid' : torch.tensor(1.0, device=device),
        'weak': torch.tensor(1.0, device=device),
        'iface':torch.tensor(1.0, device=device),
        'edi' : torch.tensor(1.0, device=device),
    }
    ema_decay = 0.98
    warmup_epochs = 5  # ramp physics λ over these epochs

    # relative weights between normalized physics terms
    w = dict(rand=1.0, mid=1.0, weak=1.0, iface=1.0, Epen=0.01)

    print("Epoch |   Time   | DataLoss | PhysLoss | TotalLoss | u_t Term | μ_spatial Term | Test relL2 | energy_loss | scheme_loss | LR")
    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        data_loss_acc = phys_loss_acc = total_loss_acc = 0.0
        energy_loss_acc = 0.0
        ut_mse_acc = mu_mse_acc = 0.0
        n_batches = 0

        # physics λ ramp (keeps pure data-driven stable early on)
        lam = pde_weight_target * min(1.0, (ep + 1) / max(1, warmup_epochs))

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)

            optimizer.zero_grad(set_to_none=True)

            # --- PURE DATA-DRIVEN PATH (no physics computed at all) ---
            if lam == 0.0:
                if amp_allowed:
                    with autocast('cuda', enabled=True):
                        y_pred = model(x)
                        loss_data = F.mse_loss(y_pred, y)
                        loss_total = loss_data
                    scaler.scale(loss_total).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    y_pred = model(x)
                    loss_data = F.mse_loss(y_pred, y)
                    loss_total = loss_data
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # logging placeholders
                loss_phys = torch.tensor(0.0, device=device)
                loss_Epen = torch.tensor(0.0, device=device)

            else:
                # --- HYBRID PATH (physics enabled) ---
                if amp_allowed:
                    with autocast('cuda', enabled=True):
                        y_pred = model(x)
                        loss_data = F.mse_loss(y_pred, y)

                        # raw physics heads
                        l_rand_raw = physics_residual_random_collocation(u_in_last, y_pred)
                        l_mid_raw  = physics_residual_midpoint(u_in_last, y_pred)
                        l_weak_raw = physics_residual_weak_lowk(u_in_last, y_pred, kfrac=0.25)
                        l_iface_raw= physics_residual_interface_weighted(u_in_last, y_pred, alpha=8.0, tau=None, q=0.75)
                        l_edi_raw, Ep_mean, E0_mean = energy_dissipation_identity_loss_midpoint(u_in_last, y_pred)

                        # update EMAs (stop-grad)
                        with torch.no_grad():
                            for k, v in [('rand', l_rand_raw), ('mid', l_mid_raw), ('weak', l_weak_raw), ('iface', l_iface_raw), ('edi', l_edi_raw)]:
                                ema[k] = ema_decay * ema[k] + (1-ema_decay) * (v.detach() + 1e-12)

                        # normalized physics sum
                        l_rand = l_rand_raw / ema['rand']
                        l_mid  = l_mid_raw  / ema['mid']
                        l_weak = l_weak_raw / ema['weak']
                        l_iface= l_iface_raw/ ema['iface']
                        l_edi  = l_edi_raw  / ema['edi']

                        loss_phys  = (w['rand']*l_rand + w['mid']*l_mid + w['weak']*l_weak + w['iface']*l_iface + l_edi)
                        loss_Epen  = w['Epen'] * energy_penalty(u_in_last, y_pred, config.DX, config.EPS2)

                        loss_total = (1.0 - lam) * loss_data + lam * (loss_phys + loss_Epen)

                    scaler.scale(loss_total).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer); scaler.update()

                else:
                    y_pred = model(x)
                    loss_data = F.mse_loss(y_pred, y)

                    l_rand_raw = physics_residual_random_collocation(u_in_last, y_pred)
                    l_mid_raw  = physics_residual_midpoint(u_in_last, y_pred)
                    l_weak_raw = physics_residual_weak_lowk(u_in_last, y_pred, kfrac=0.25)
                    l_iface_raw= physics_residual_interface_weighted(u_in_last, y_pred, alpha=8.0, tau=None, q=0.75)
                    l_edi_raw, Ep_mean, E0_mean = energy_dissipation_identity_loss_midpoint(u_in_last, y_pred)

                    with torch.no_grad():
                        for k, v in [('rand', l_rand_raw), ('mid', l_mid_raw), ('weak', l_weak_raw), ('iface', l_iface_raw), ('edi', l_edi_raw)]:
                            ema[k] = ema_decay * ema[k] + (1-ema_decay) * (v.detach() + 1e-12)

                    l_rand = l_rand_raw / ema['rand']
                    l_mid  = l_mid_raw  / ema['mid']
                    l_weak = l_weak_raw / ema['weak']
                    l_iface= l_iface_raw/ ema['iface']
                    l_edi  = l_edi_raw  / ema['edi']

                    loss_phys  = (w['rand']*l_rand + w['mid']*l_mid + w['weak']*l_weak + w['iface']*l_iface + l_edi)
                    loss_Epen  = w['Epen'] * energy_penalty(u_in_last, y_pred, config.DX, config.EPS2)

                    loss_total = (1.0 - lam) * loss_data + lam * (loss_phys + loss_Epen)

                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

            # --- always compute debug terms from current batch prediction ---
            debug_ut, debug_mu = _debug_terms(u_in_last, y_pred)

            # --- accumulators ---
            data_loss_acc   += float(loss_data.detach().cpu())
            phys_loss_acc   += float(loss_phys.detach().cpu())
            energy_loss_acc += float(loss_Epen.detach().cpu())
            total_loss_acc  += float(loss_total.detach().cpu())
            ut_mse_acc      += float(debug_ut.detach().cpu())
            mu_mse_acc      += float(debug_mu.detach().cpu())
            n_batches       += 1

        # --- Eval ---
        model.eval()
        with torch.no_grad():
            rels = []
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if amp_allowed:
                    with autocast('cuda', enabled=True):
                        y_pred = model(x)
                else:
                    y_pred = model(x)
                rels.append(relative_l2(y_pred, y))
            test_rel = torch.cat(rels, dim=0).mean().item()

        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']
        print(f"{ep:5d} | {t2-t1:7.3f} | "
              f"{data_loss_acc/n_batches:8.3e} | {phys_loss_acc/n_batches:8.3e} | {total_loss_acc/n_batches:9.3e} | "
              f"{(ut_mse_acc/n_batches):8.3e} | {(mu_mse_acc/n_batches):14.3e} | "
              f"{test_rel:10.3e} |  {energy_loss_acc:10.3e} |   0.000e+00 | {lr: .2e}")


# ---------------------
# Optional: simple rollout/eval helpers (data-driven rollout)
# ---------------------
def rollout_autoregressive(model, traj_np, T_in, Nt=100):
    """
    traj_np: (Nt+1, S,S,S) ground truth trajectory for one sample
    returns pred: same shape, with pred[0:T_in]=gt[0:T_in], rest autoregressive
    (Pure model rollout, no solver-in-the-loop.)
    """
    device = next(model.parameters()).device
    pred = np.zeros_like(traj_np, dtype=np.float32)
    pred[:T_in] = traj_np[:T_in]
    for t in range(T_in-1, Nt):
        x = torch.from_numpy(pred[t-(T_in-1):t+1]).permute(1, 2, 3, 0).unsqueeze(0).to(device)  # (1,S,S,S,T_in)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.USE_AMP):
            y_hat = model(x).squeeze(0).squeeze(-1).detach().cpu().numpy()
        pred[t+1] = y_hat
    return pred


def relative_l2_scalar(a, b, eps=1e-12):
    num = np.linalg.norm(a.ravel() - b.ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return num / den


def evaluate_stats_and_plot(model, mat_path, test_ids, times):
    """
    Utility for offline evaluation and visualization (central z-slice).
    """
    import matplotlib.pyplot as plt

    with h5py.File(mat_path, "r") as f:
        dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
        Nz, Ny, Nx, Nt, Ns = dset.shape
        assert Nt == config.SAVED_STEPS

        rel_errors = {t: [] for t in times}

        # pick first test id for plotting
        pid = int(test_ids[0])
        gt_raw = np.array(dset[:, :, :, :, pid], dtype=np.float32)  # (Nz,Ny,Nx,Nt)
        gt = np.transpose(gt_raw, (3, 2, 1, 0))                      # (Nt,Nx,Ny,Nz)
        pred = rollout_autoregressive(model, gt, config.T_IN_CHANNELS,
                                      Nt=config.TOTAL_TIME_STEPS)

        # stats across test ids
        for sid in test_ids:
            gt_raw = np.array(dset[:, :, :, :, sid], dtype=np.float32)
            gt_s = np.transpose(gt_raw, (3, 2, 1, 0))
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

        # plots (central z-slice)
        S = gt.shape[1]; zc = S // 2
        fig, axes = plt.subplots(3, len(times), figsize=(4*len(times), 9))
        for j, t in enumerate(times):
            exact = gt[t, :, :, zc]
            predt = pred[t, :, :, zc]
            rel   = np.abs(predt - exact) / (np.abs(exact) + 1e-8)

            im0 = axes[0, j].imshow(exact, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, j].set_title(f"Exact t={t}");  fig.colorbar(im0, ax=axes[0, j], shrink=0.8)
            im1 = axes[1, j].imshow(predt, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, j].set_title(f"Pred t={t}");   fig.colorbar(im1, ax=axes[1, j], shrink=0.8)
            im2 = axes[2, j].imshow(rel, origin='lower', cmap='viridis')
            axes[2, j].set_title(f"Rel. L2 (px) t={t}"); fig.colorbar(im2, ax=axes[2, j], shrink=0.8)

            for r in range(3):
                axes[r, j].set_xlabel('x'); axes[r, j].set_ylabel('y')
        plt.tight_layout(); plt.show()
