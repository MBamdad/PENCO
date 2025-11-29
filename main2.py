#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian PINO for 1D viscous Burgers (periodic) — fixed, stable dataset generator
- Stable spectral Burgers solver in Fourier space with 2/3 de-aliasing
- CFL-aware substepping inside each output interval
- Variational Bayes (factorized Gaussian) conv layers (Bayes-by-Backprop)
- Physics-informed ELBO: data NLL + physics residual + IC + KL
- Evaluation via relative L2 error against exact solution
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# -----------------------------
# Repro & device
# -----------------------------
SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PDE: Stable 1D viscous Burgers solver (periodic) in Fourier space + RK4
# -----------------------------
def burgers_spectral_rk4_stable(u0, nu, nx=256, nt=64, T=1.0, cfl=0.4, c_visc=0.4):
    """
    u_t + u u_x = nu u_xx  on x in [0, 2pi), t in [0, T]
    Periodic BCs. Work in Fourier space:
        d/dt u_hat = -0.5 * i*k * FFT(u^2) - nu * k^2 * u_hat
    Use 2/3 de-aliasing on FFT(u^2).
    Output nt snapshots including t=0 and t=T (uniformly spaced).
    The integrator substeps adaptively (CFL for convection & diffusion).
    """
    # Grid & wavenumbers
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    dx = x[1] - x[0]
    # numpy's fftfreq gives cycles per unit; multiply by 2π to get angular wavenumber
    k = 2*np.pi * np.fft.fftfreq(nx, d=dx)                    # shape (nx,)
    ik = 1j * k
    k2 = k**2

    # 2/3 de-aliasing mask in FFT index order: keep modes |m| <= N/3
    m = np.fft.fftfreq(nx) * nx                               # integer mode numbers
    dealias_mask = (np.abs(m) <= (nx // 3)).astype(np.float64)

    # set outputs
    t_out = np.linspace(0.0, T, nt, endpoint=True)
    out = np.zeros((nt, nx), dtype=np.float64)
    out[0] = u0

    # Initial spectral state
    u_hat = np.fft.fft(u0)

    def rhs(u_hat_local):
        # inverse to physical space
        u = np.fft.ifft(u_hat_local).real
        # compute u^2 in physical, FFT, de-alias
        u2_hat = np.fft.fft(u*u)
        u2_hat *= dealias_mask
        # RHS in spectral
        return -0.5 * ik * u2_hat - nu * (k2 * u_hat_local)

    # Integrate with adaptive substeps to each output time
    t_curr = 0.0
    idx_out = 1
    for n in range(1, nt):
        t_target = t_out[n]
        while t_curr < t_target - 1e-15:
            # recompute physical max |u| for CFL
            u_phys = np.fft.ifft(u_hat).real
            umax = max(1e-6, float(np.max(np.abs(u_phys))))
            # convection CFL: dt <= cfl * dx / umax
            dt_cfl = cfl * dx / umax
            # diffusion stability: dt <= c_visc * dx^2 / nu  (heuristic)
            if nu > 0:
                dt_visc = c_visc * dx*dx / nu
                dt_sub = min(dt_cfl, dt_visc, t_target - t_curr)
            else:
                dt_sub = min(dt_cfl, t_target - t_curr)

            # One RK4 step in spectral variables
            k1 = rhs(u_hat)
            k2 = rhs(u_hat + 0.5*dt_sub*k1)
            k3 = rhs(u_hat + 0.5*dt_sub*k2)
            k4 = rhs(u_hat + dt_sub*k3)
            u_hat = u_hat + (dt_sub/6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # (optional) small spectral filter to counter roundoff at the tail
            u_hat *= (1.0 - 1e-12 * k2)

            t_curr += dt_sub

        # store snapshot
        out[n] = np.fft.ifft(u_hat).real

    return out  # shape (nt, nx)

# -----------------------------
# Dataset generation
# -----------------------------
@dataclass
class BurgersConfig:
    nx: int = 256
    nt: int = 64
    T: float = 1.0
    nu: float = 0.01
    n_train: int = 200
    n_val: int = 40
    n_test: int = 40
    sensors_frac: float = 0.05  # fraction of spatiotemporal points observed for data term
    # stability knobs (passed to solver)
    cfl: float = 0.4
    c_visc: float = 0.4

def sample_initial_condition(nx):
    """
    Random smooth initial condition as a sum of sines/cosines with random amplitudes and phases.
    """
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    u0 = np.zeros_like(x)
    max_mode = 6
    for m in range(1, max_mode+1):
        amp = np.random.randn() * np.exp(-0.4*m)  # decay with m
        phase = 2*np.pi*np.random.rand()
        if np.random.rand() < 0.5:
            u0 += amp * np.sin(m*x + phase)
        else:
            u0 += amp * np.cos(m*x + phase)
    # normalize to a moderate range
    u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
    return u0.astype(np.float64)

def make_dataset(cfg: BurgersConfig):
    nx, nt, T, nu = cfg.nx, cfg.nt, cfg.T, cfg.nu
    def gen_one():
        u0 = sample_initial_condition(nx)
        u = burgers_spectral_rk4_stable(
            u0, nu, nx=nx, nt=nt, T=T, cfl=cfg.cfl, c_visc=cfg.c_visc
        )  # (nt, nx)
        # safety: replace any stray NaN/inf (shouldn't happen now)
        if not np.isfinite(u).all():
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        return u0, u

    train = [gen_one() for _ in range(cfg.n_train)]
    val   = [gen_one() for _ in range(cfg.n_val)]
    test  = [gen_one() for _ in range(cfg.n_test)]

    # sensor indices for data likelihood
    grid_N = nx * nt
    k_obs = max(1, int(cfg.sensors_frac * grid_N))
    obs_idx = np.random.choice(grid_N, size=k_obs, replace=False)
    obs_t_idx = obs_idx // nx
    obs_x_idx = obs_idx % nx
    sensors = (obs_t_idx.astype(np.int64), obs_x_idx.astype(np.int64))

    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    t = np.linspace(0, T, nt, endpoint=True)
    return train, val, test, sensors, x, t

# -----------------------------
# Bayesian layers (factorized Gaussian) and KL
# -----------------------------
class BayesConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 prior_std=0.1):
        super().__init__()
        self.ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride, self.padding = stride, padding
        # variational params
        self.W_mu = nn.Parameter(torch.randn(out_ch, in_ch, *self.ks) * 0.02)
        self.W_rho = nn.Parameter(torch.full((out_ch, in_ch, *self.ks), -5.0))
        self.b_mu = nn.Parameter(torch.zeros(out_ch))
        self.b_rho = nn.Parameter(torch.full((out_ch,), -5.0))
        # prior
        self.prior_var = prior_std**2

    def kl_term(self):
        W_sigma = torch.log1p(torch.exp(self.W_rho))  # softplus
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        kl_W = 0.5 * torch.sum(
            (W_sigma**2 + self.W_mu**2)/self.prior_var - 1.0 - 2.0*torch.log(W_sigma) + math.log(self.prior_var)
        )
        kl_b = 0.5 * torch.sum(
            (b_sigma**2 + self.b_mu**2)/self.prior_var - 1.0 - 2.0*torch.log(b_sigma) + math.log(self.prior_var)
        )
        return kl_W + kl_b

    def forward(self, x):
        eps_W = torch.randn_like(self.W_mu)
        eps_b = torch.randn_like(self.b_mu)
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        W = self.W_mu + W_sigma * eps_W
        b = self.b_mu + b_sigma * eps_b
        return F.conv2d(x, W, b, stride=self.stride, padding=self.padding), self.kl_term()

class BayesLinear(nn.Module):
    def __init__(self, in_f, out_f, prior_std=0.1):
        super().__init__()
        self.W_mu = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.W_rho = nn.Parameter(torch.full((out_f, in_f), -5.0))
        self.b_mu = nn.Parameter(torch.zeros(out_f))
        self.b_rho = nn.Parameter(torch.full((out_f,), -5.0))
        self.prior_var = prior_std**2

    def kl_term(self):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        kl_W = 0.5 * torch.sum(
            (W_sigma**2 + self.W_mu**2)/self.prior_var - 1.0 - 2.0*torch.log(W_sigma) + math.log(self.prior_var)
        )
        kl_b = 0.5 * torch.sum(
            (b_sigma**2 + self.b_mu**2)/self.prior_var - 1.0 - 2.0*torch.log(b_sigma) + math.log(self.prior_var)
        )
        return kl_W + kl_b

    def forward(self, x):
        eps_W = torch.randn_like(self.W_mu)
        eps_b = torch.randn_like(self.b_mu)
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        W = self.W_mu + W_sigma * eps_W
        b = self.b_mu + b_sigma * eps_b
        return F.linear(x, W, b), self.kl_term()

# -----------------------------
# Bayesian PINO model (2D CNN over (t,x))
# Input: channels = 2  (u0 tiled over t, and time grid channel)
# Output: u_hat(t,x) as 1 channel
# -----------------------------
class BayesianPINO(nn.Module):
    def __init__(self, hidden=64, prior_std=0.1):
        super().__init__()
        self.c1 = BayesConv2d(2, hidden, 3, padding=1, prior_std=prior_std)
        self.c2 = BayesConv2d(hidden, hidden, 3, padding=1, prior_std=prior_std)
        self.c3 = BayesConv2d(hidden, hidden, 3, padding=1, prior_std=prior_std)
        self.c4 = BayesConv2d(hidden, 1, 1, padding=0, prior_std=prior_std)
        self.act = nn.SiLU()
        # learnable observation noise std
        self.log_sigma_y = nn.Parameter(torch.tensor(-2.0))

    def forward(self, u0_batch, tgrid):
        """
        u0_batch: (B, nx)
        tgrid: (nt,) monotone ascending
        Returns: u_hat: (B, nt, nx), total_kl (scalar), sigma_y (scalar+)
        """
        B, nx = u0_batch.shape
        nt = tgrid.shape[0]
        # build input tensor: (B, 2, nt, nx)
        u0_tiled = u0_batch[:, None, None, :].repeat(1, 1, nt, 1)
        tchan = tgrid[None, None, :, None].repeat(B, 1, 1, nx)
        x = torch.cat([u0_tiled, tchan], dim=1)

        kls = []
        x, kl = self.c1(x); kls.append(kl); x = self.act(x)
        x, kl = self.c2(x); kls.append(kl); x = self.act(x)
        x, kl = self.c3(x); kls.append(kl); x = self.act(x)
        x, kl = self.c4(x); kls.append(kl)
        u_hat = x[:, 0]  # (B, nt, nx)
        total_kl = torch.stack(kls).sum()
        sigma_y = torch.nn.functional.softplus(self.log_sigma_y) + 1e-6
        return u_hat, total_kl, sigma_y

# -----------------------------
# Physics residual (finite differences on uniform grid)
# -----------------------------
def periodic_diff_x(u, dx):
    return (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2.0*dx)

def periodic_laplace_x(u, dx):
    return (torch.roll(u, shifts=-1, dims=-1) - 2.0*u + torch.roll(u, shifts=1, dims=-1)) / (dx*dx)

def forward_time_diff(u, dt):
    return (u[:, 1:, :] - u[:, :-1, :]) / dt

def physics_residual(u_hat, nu, dx, dt):
    """
    r = u_t + u u_x - nu u_xx  over 0..nt-2 (forward dt)
    """
    ut = forward_time_diff(u_hat, dt)  # (B, nt-1, nx)
    uh_mid = u_hat[:, :-1, :]
    ux = periodic_diff_x(uh_mid, dx)
    uxx = periodic_laplace_x(uh_mid, dx)
    r = ut + uh_mid * ux - nu * uxx
    return r

# -----------------------------
# Data utilities
# -----------------------------
def tensorize_dataset(data_list, device):
    u0s = np.stack([d[0] for d in data_list], axis=0)    # (N, nx)
    us  = np.stack([d[1] for d in data_list], axis=0)    # (N, nt, nx)
    u0s = torch.tensor(u0s, dtype=torch.float32, device=device)
    us  = torch.tensor(us,  dtype=torch.float32, device=device)
    return u0s, us

def make_observations(u_exact, sensors):
    t_idx, x_idx = sensors
    t_idx = torch.tensor(t_idx, dtype=torch.long, device=u_exact.device)
    x_idx = torch.tensor(x_idx, dtype=torch.long, device=u_exact.device)
    y = u_exact[:, t_idx, x_idx]  # (B, K)
    return y, t_idx, x_idx

# -----------------------------
# Training & evaluation
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 10
    lr: float = 3e-3
    beta_kl_final: float = 1.0   # anneal 0->final over first 30% epochs
    lambda_phys: float = 5.0
    lambda_ic: float = 1.0
    mc_samples: int = 2          # # samples of weights per batch for ELBO expectation

def relative_l2(pred, truth, eps=1e-12):
    num = torch.norm((pred - truth).reshape(pred.shape[0], -1), dim=1)
    den = torch.norm(truth.reshape(pred.shape[0], -1), dim=1) + eps
    return (num/den).mean().item()

def train_and_eval():
    # Configs
    dcfg = BurgersConfig()
    tcfg = TrainConfig()
    print("Device:", DEVICE)

    # Dataset
    print("Generating dataset...")
    train_raw, val_raw, test_raw, sensors, x, t = make_dataset(dcfg)
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    # Tensorize
    u0_train, u_train = tensorize_dataset(train_raw, DEVICE)
    u0_val,   u_val   = tensorize_dataset(val_raw, DEVICE)
    u0_test,  u_test  = tensorize_dataset(test_raw, DEVICE)

    Ntr = u0_train.shape[0]
    Nva = u0_val.shape[0]
    Nte = u0_test.shape[0]
    print(f"Train/Val/Test sizes: {Ntr}/{Nva}/{Nte}")
    tgrid = torch.tensor(t, dtype=torch.float32, device=DEVICE)  # (nt,)
    nu = torch.tensor(dcfg.nu, dtype=torch.float32, device=DEVICE)

    # Model & optim
    model = BayesianPINO(hidden=64, prior_std=0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    def get_batches(N, bs):
        idx = torch.randperm(N)
        for i in range(0, N, bs):
            yield idx[i:i+bs]

    best_val = float('inf')
    for epoch in range(1, tcfg.epochs+1):
        model.train()
        train_loss = 0.0
        # KL annealing
        frac = min(1.0, epoch / max(1, int(0.3 * tcfg.epochs)))
        beta_kl = tcfg.beta_kl_final * frac

        for batch_idx in get_batches(Ntr, tcfg.batch_size):
            opt.zero_grad()
            u0_b = u0_train[batch_idx]
            u_b  = u_train[batch_idx]

            # data observations
            y_obs, t_idx, x_idx = make_observations(u_b, sensors)  # (B, K)

            elbo_terms = []
            kl_terms = []
            for _ in range(tcfg.mc_samples):
                u_hat, kl, sigma_y = model(u0_b, tgrid)            # (B, nt, nx), scalar, scalar
                # Data NLL at sensors: Gaussian
                y_pred = u_hat[:, t_idx, x_idx]                    # (B, K)
                resid_y = y_pred - y_obs
                nll = 0.5 * torch.mean((resid_y / sigma_y)**2) + torch.log(sigma_y)
                # Physics residual over all internal points
                r = physics_residual(u_hat, nu, dx, dt)            # (B, nt-1, nx)
                phys = 0.5 * tcfg.lambda_phys * torch.mean(r**2)
                # IC penalty at t=0
                ic = 0.5 * tcfg.lambda_ic * torch.mean((u_hat[:, 0, :] - u0_b)**2)
                loss_sample = nll + phys + ic
                elbo_terms.append(loss_sample)
                kl_terms.append(kl)

            loss_mc = torch.stack(elbo_terms).mean()
            kl_mc = torch.stack(kl_terms).mean()
            loss = loss_mc + beta_kl * (kl_mc / (u0_b.shape[0]))  # per-sample scale

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * u0_b.shape[0]

        train_loss /= Ntr

        # Validation
        model.eval()
        with torch.no_grad():
            val_err = 0.0
            cnt = 0
            for batch_idx in get_batches(Nva, tcfg.batch_size):
                u0_b = u0_val[batch_idx]
                u_b  = u_val[batch_idx]
                S_eval = 4
                preds = []
                for _ in range(S_eval):
                    u_hat, _, _ = model(u0_b, tgrid)
                    preds.append(u_hat)
                u_mean = torch.stack(preds, dim=0).mean(dim=0)  # (B, nt, nx)
                num = torch.norm((u_mean - u_b).reshape(u_b.shape[0], -1), dim=1)
                den = torch.norm(u_b.reshape(u_b.shape[0], -1), dim=1) + 1e-12
                val_err += torch.sum(num/den).item()
                cnt += u_b.shape[0]
            val_rel_l2 = val_err / cnt

        if val_rel_l2 < best_val:
            best_val = val_rel_l2
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train_loss {train_loss:.6f} | val_relL2 {val_rel_l2:.6f} | best {best_val:.6f} | betaKL {beta_kl:.2f}")

    # -----------------------------
    # Final test evaluation
    # -----------------------------
    model.eval()
    with torch.no_grad():
        batch_size = tcfg.batch_size
        test_rel_l2 = 0.0
        test_ale = 0.0
        test_epistemic = 0.0
        cnt = 0
        for i0 in range(0, Nte, batch_size):
            sel = slice(i0, min(Nte, i0+batch_size))
            u0_b = u0_test[sel]
            u_b  = u_test[sel]
            # Ensemble for predictive mean and variance
            S = 16
            preds = []
            for _ in range(S):
                u_hat, _, _ = model(u0_b, tgrid)
                preds.append(u_hat)
            U = torch.stack(preds, dim=0)              # (S, B, nt, nx)
            u_mean = U.mean(dim=0)                     # (B, nt, nx)
            epistemic_var = U.var(dim=0, unbiased=False)  # (B, nt, nx)
            # relative L2
            num = torch.norm((u_mean - u_b).reshape(u_b.shape[0], -1), dim=1)
            den = torch.norm(u_b.reshape(u_b.shape[0], -1), dim=1) + 1e-12
            test_rel_l2 += torch.sum(num/den).item()
            # uncertainty diagnostics (mean over grid)
            test_epistemic += epistemic_var.mean(dim=(1,2)).sum().item()
            sigma_y = (F.softplus(model.log_sigma_y) + 1e-6).item()
            test_ale += (sigma_y**2) * u_b.shape[0]
            cnt += u_b.shape[0]

        test_rel_l2 /= cnt
        test_epistemic /= cnt
        test_ale /= cnt

    print("\n=== Test metrics ===")
    print(f"Relative L2 error (mean pred vs exact): {test_rel_l2:.4f}")
    print(f"Predictive epistemic variance (avg over grid): {test_epistemic:.6f}")
    print(f"Aleatoric variance (learned sigma_y^2): {test_ale:.6f}")

if __name__ == "__main__":
    train_and_eval()
