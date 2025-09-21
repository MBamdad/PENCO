import h5py, numpy as np, torch, random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import config
import math
from functions import (
    mu_ac, physics_residual_midpoint, scheme_residual_fourier, loss_mm_projection,
    semi_implicit_step, energy_penalty, physics_residual_normalized,
    physics_guided_update_ch , hminus1_mse ,physics_residual_midpoint_normalized_ch_direct ,
    physics_guided_update_ch_optimal
)

import matplotlib
matplotlib.use('TkAgg')


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

    # --- read Ns from the file (last dim of phi: (Nz,Ny,Nx,Nt,Ns)) ---
    with h5py.File(config.MAT_DATA_PATH, "r") as f:
        Ns = int(f["phi"].shape[-1])

    # ids = 0..Ns-1, shuffled
    all_ids = np.arange(Ns)
    rng.shuffle(all_ids)

    # --- clamp splits so they sum to Ns and never go negative ---
    n_train_req = int(config.N_TRAIN)
    n_test_req  = int(config.N_TEST)

    n_train = min(max(0, n_train_req), Ns)                 # 0..Ns
    # try to keep requested test size, but not beyond remaining
    n_test  = min(max(1, n_test_req), max(0, Ns - n_train))  # at least 1 if possible
    # if there isn't room for a test set (e.g., Ns==1), force n_test=0 and shrink n_train
    if n_test == 0 and Ns >= 1:
        n_train = max(0, Ns - 1)
        n_test  = 1

    n_unused = Ns - n_train - n_test
    assert n_unused >= 0, "Split sizes exceed dataset size."

    # build the base dataset over ALL ids so RAM loader knows Ns
    base = AC3DTrajectoryDataset(config.MAT_DATA_PATH, all_ids)

    # split into train/test/unused
    train_dataset, test_dataset, _ = random_split(
        base, [n_train, n_test, n_unused],
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

    # deterministic list of test IDs (after the train split)
    test_indices = all_ids[n_train:n_train + n_test].tolist()

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
## Correct
def train_fno_hybrid(model, train_loader, test_loader, optimizer, scheduler, device, pde_weight=None):
    pde_weight = config.PDE_WEIGHT if pde_weight is None else pde_weight

    # physics term weights (baseline-compatible)
    USE_CH = (config.PROBLEM == 'CH3D')
    w_mid, w_fft = 1.0, 1.0
    w_mm = 0.0 if USE_CH else 1.0  # CH: disable MM; AC: unchanged
    MID_NORM_MIX = 0.5
    CLIP_NORM = 1.0

    print("Epoch |   Time   | DataLoss | PhysLoss | TotalLoss | Test relL2 | energy_loss | scheme_loss | l_mid_norm_ch_cc | LR")
    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        data_loss_acc = phys_loss_acc = total_loss_acc = l_mid_norm_ch_cc = 0.0
        energy_loss_acc = scheme_loss_acc = 0.0
        ut_mse_acc = mu_mse_acc = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)

            optimizer.zero_grad(set_to_none=True)

            # forward
            y_pred = model(x)
            y_hat = y_pred


            # ---- data term (keep your scaling so behavior remains stable) ----
            if pde_weight == 0:
                loss_data = 1.0 * F.mse_loss(y_pred, y)
            else:
                loss_data = 1e2 * F.mse_loss(y_pred, y)

            # ----- physics bundle -----
            if USE_CH:
                l_fft = scheme_residual_fourier(u_in_last, y_hat)
                l_mid_norm = physics_residual_midpoint_normalized_ch_direct(u_in_last, y_hat)

                # teacher (unchanged)
                u_si1 = semi_implicit_step(u_in_last, config.DT, config.DX, config.EPS2)
                loss_scheme1 = F.mse_loss(y_hat, u_si1)

                with torch.no_grad():
                    u_si2 = semi_implicit_step(u_si1, config.DT, config.DX, config.EPS2)
                x2 = torch.cat([x[..., 1:], y_hat], dim=-1)  # <-- use y_hat
                y_hat2 = model(x2)
                # also pass through PGU for the second step so the scheme loss sees physics-consistent outputs
                if USE_CH:
                    y_hat2 = physics_guided_update_ch(y_hat, y_hat2, alpha=0.3, tau=0.3)
                loss_scheme2 = F.mse_loss(y_hat2, u_si2)
                loss_scheme = 0.7 * loss_scheme1 + 0.3 * loss_scheme2

                up = y_hat.squeeze(-1)
                l_teacher_hm1 = hminus1_mse(up, u_si1.squeeze(-1), config.DX)

                loss_phys = 4e-2 * (
                        1e-1 * l_fft +
                        2e-2 * l_mid_norm +
                        5e-4 * l_teacher_hm1
                )
            else:
                # (your existing non-CH path unchanged)
                l_mid_plain = physics_residual_midpoint(u_in_last, y_pred)
                l_mid_norm, _, _ = physics_residual_normalized(u_in_last, y_pred)
                l_mid = 0.5 * l_mid_plain + 0.5 * l_mid_norm
                l_fft = scheme_residual_fourier(u_in_last, y_pred)
                l_mm, l_proj_dbg, l_stat_dbg = loss_mm_projection(u_in_last, y_pred)

                loss_phys = 1e-5 * (w_mid * l_mid + w_fft * l_fft + w_mm * l_mm)

                u_si = semi_implicit_step(u_in_last, config.DT, config.DX, config.EPS2)
                loss_scheme = F.mse_loss(y_pred, u_si)

            loss_energy = 0.05 * energy_penalty(u_in_last, y_pred, config.DX, config.EPS2)

            # ---- total loss (unchanged structure) ----
            loss_total = (1.0 - pde_weight) * loss_data + pde_weight * (loss_phys + loss_scheme + loss_energy)

            # backward
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()


            # accumulators
            data_loss_acc  += loss_data.item()
            phys_loss_acc  += loss_phys.item()
            l_mid_norm_ch_cc += l_mid_norm.item()
            energy_loss_acc += loss_energy.item()
            scheme_loss_acc += loss_scheme.item()
            total_loss_acc += loss_total.item()
            n_batches      += 1

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

        scheduler.step()
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
            if config.PROBLEM == 'CH3D':
                y_next = physics_guided_update_ch_optimal(x[..., -1:], y_next, tau=0.3, alpha_cap=0.6)
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

    with h5py.File(mat_path, "r") as f:
        dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
        Nz, Ny, Nx, Nt, Ns = dset.shape
        assert Nt == config.SAVED_STEPS

        rel_errors = {t: [] for t in times}

        # pick first test id for plotting
        pid = int(test_ids[0])
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

            im0 = axes[0, j].imshow(exact, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, j].set_title(f"Exact t={t}");  fig.colorbar(im0, ax=axes[0, j], shrink=0.8)
            im1 = axes[1, j].imshow(predt, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, j].set_title(f"Pred t={t}");   fig.colorbar(im1, ax=axes[1, j], shrink=0.8)
            im2 = axes[2, j].imshow(rel, origin='lower', cmap='viridis')
            axes[2, j].set_title(f"Rel. L2 (px) t={t}"); fig.colorbar(im2, ax=axes[2, j], shrink=0.8)

            for r in range(3):
                axes[r, j].set_xlabel('x'); axes[r, j].set_ylabel('y')
        plt.tight_layout(); plt.show()
