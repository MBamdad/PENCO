import torch
import torch.nn.functional as F
import numpy as np
import config

# -------------------------
# FFT helpers and operators
# -------------------------
@torch.no_grad()
def _fft_wavenumbers_3d(nx, ny, nz, dx):
    kx = torch.fft.fftfreq(nx, d=dx)
    ky = torch.fft.fftfreq(ny, d=dx)
    kz = torch.fft.fftfreq(nz, d=dx)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    minus_k2 = -((2*np.pi)**2) * (kx**2 + ky**2 + kz**2)
    return minus_k2

# add near your other helpers
def _k_spectrum(nx, ny, nz, dx, device, dtype=torch.float32):
    kx = torch.fft.fftfreq(nx, d=dx).to(device)
    ky = torch.fft.fftfreq(ny, d=dx).to(device)
    kz = torch.fft.fftfreq(nz, d=dx).to(device)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k2 = (2*np.pi)**2 * (kx**2 + ky**2 + kz**2)
    return k2.to(dtype)

# ---- existing k-spectrum helper is already present ----
# def _k_spectrum(...)

import torch, math
import torch.fft
import torch.nn.functional as F
import numpy as np
import config



def _fft_rk2(nx, ny, nz, dx, device, dtype=torch.float32):
    """Return k^2 (=(2π)^2 |k|^2) and a safe inverse (1/(k^2+ε0)) for H^{-1} norm."""
    k2 = _k_spectrum(nx, ny, nz, dx, device, dtype=dtype)  # (S,S,S)
    # small floor for k=0 to avoid division by zero; choose physical smallest mode ~ (2π/L)^2
    # L = S*dx; minimal nonzero k^2 is about (2π/L)^2. Use a small fraction of that as ε0.
    L = nx * dx
    eps_k2 = (2*np.pi / L)**2 * 1e-2
    inv_k2_safe = 1.0 / (k2 + eps_k2)
    return k2, inv_k2_safe

def _charbonnier_mean(x, eps=1e-8):
    # Robust L2 ~1 when x small, ~|x| when large; stabilizes max spikes.
    return torch.sqrt(x*x + eps).mean()

# === NEW: L2 Gauss–Lobatto collocation for AC3D (identical form to SH/PFC/MBE) ===
import math


def physics_collocation_tau_L2_AC(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                  normalize=True):
    """
    AC3D collocation at u_tau:
      R_tau = (u^{n+1}-u^n)/dt - RHS_AC(u_tau), scored in L2,
      with optional per-sample normalization (same as SH/PFC/MBE).
    """
    assert config.PROBLEM == 'AC3D'
    dt, dx = config.DT, config.DX

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float()
    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    rhs_tau = pde_rhs(u_tau, dx, config.EPSILON_PARAM)  # -> AC RHS under AC3D

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau
    return (R**2).mean().to(u_pred.dtype)

def physics_collocation_tau_L2_AC_Tout(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                  normalize=True):
    """
    AC3D collocation at u_tau:
      R_tau = (u^{n+1}-u^n)/dt - RHS_AC(u_tau), scored in L2.

    Extended to handle:
      u_pred: (B,S,S,S,1)  or  (B,S,S,S,T_out)

    For T_out > 1, we interpret u_pred[...,k] as u^{n+k} and build
    a temporal chain u^n, u^{n+1}, ..., u^{n+T_out}, computing
    residuals for each step.
    """
    assert config.PROBLEM == 'AC3D'
    dt, dx = config.DT, config.DX

    # u0: last known state, shape (B,S,S,S)
    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)

    up = u_pred.float()
    # Case 1: original single-step case
    if up.shape[-1] == 1:
        up = up.squeeze(-1)  # (B,S,S,S)

        ut = (up - u0) / dt
        u_tau = (1.0 - tau) * u0 + tau * up     # (B,S,S,S)

        rhs_tau = pde_rhs(u_tau, dx, config.EPSILON_PARAM)  # (B,S,S,S)

        if normalize:
            s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau

    # Case 2: multi-step prediction, up: (B,S,S,S,T_out)
    else:
        B, Sx, Sy, Sz, T = up.shape

        # Build the chain [u^n, u^{n+1}, ..., u^{n+T}]
        u0_exp = u0.unsqueeze(-1)                  # (B,Sx,Sy,Sz,1)
        u_all = torch.cat([u0_exp, up], dim=-1)    # (B,Sx,Sy,Sz,T+1)

        # Differences: u^{k} - u^{k-1} for k=1..T
        u_prev = u_all[..., :-1]   # (B,Sx,Sy,Sz,T)
        u_next = u_all[...,  1:]   # (B,Sx,Sy,Sz,T)

        ut = (u_next - u_prev) / dt               # (B,Sx,Sy,Sz,T)
        u_tau = (1.0 - tau) * u_prev + tau * u_next   # (B,Sx,Sy,Sz,T)

        # Flatten time into batch for pde_rhs
        u_tau_flat = u_tau.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
        rhs_tau_flat = pde_rhs(u_tau_flat, dx, config.EPSILON_PARAM)  # (B*T,Sx,Sy,Sz)
        rhs_tau = rhs_tau_flat.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)
        # rhs_tau: (B,Sx,Sy,Sz,T)

        if normalize:
            s_t = ut.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau

    return (R**2).mean().to(u_pred.dtype)


# === NEW: L2 Gauss–Lobatto collocation for CH3D (to match SH/PFC/MBE exactly) ===

def spec_high_energy(u, frac_cut=0.6):
    """
    Tiny CH-only regularizer: penalize energy in the top (1-frac_cut) fraction of modes.
    u: (B,S,S,S)
    """
    B, S, _, _ = u.shape
    uhat = torch.fft.rfftn(u, dim=(1,2,3))
    # radial mask in index space
    kx = torch.fft.fftfreq(S, d=1.0).to(u.device)
    ky = torch.fft.fftfreq(S, d=1.0).to(u.device)
    kz = torch.fft.rfftfreq(S, d=1.0).to(u.device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    r = torch.sqrt(KX*KX + KY*KY + KZ*KZ)
    rmax = r.max()
    mask = (r > frac_cut * rmax).float()
    energy = (uhat.real**2 + uhat.imag**2) * mask
    return energy.mean().to(u.dtype)



import torch
import torch.fft as tfft
import math

def _kx_grid(S, dx, device, dtype):
    # physical-size L = S*dx; match MATLAB’s 2π/L * [0..S/2, -S/2+1..-1]
    L = S * dx
    half = S // 2
    kvec = torch.cat([
        torch.arange(0, half + 1, device=device),
        torch.arange(-half + 1, 0, device=device)
    ], dim=0).to(dtype)
    return (2.0 * math.pi / L) * kvec  # (S,)
'''''
def semi_implicit_step_sh(u_in, dt, dx, eps):
    """
    One semi-implicit Swift–Hohenberg step that mirrors your MATLAB:
      s_hat = FFT(u/dt) - FFT(u^3) + 2 * (kxx+kyy+kzz) * FFT(u)
      v_hat = s_hat / (1/dt + (1 - eps) + (kxx+kyy+kzz)^2)
      u^{n+1} = IFFT(v_hat)
    Inputs:
      u_in: (B,S,S,S,1)  real
      dt, dx: scalars (float)
      eps: your config.EPSILON_PARAM (same as MATLAB 'epsilon', not eps^2)
    Returns:
      (B,S,S,S,1)
    """
    assert u_in.ndim == 5 and u_in.shape[-1] == 1
    u = u_in.squeeze(-1)

    B, Sx, Sy, Sz = u.shape
    device, dtype = u.device, u.dtype

    kx = _kx_grid(Sx, dx, device, dtype)
    ky = _kx_grid(Sy, dx, device, dtype)
    kz = _kx_grid(Sz, dx, device, dtype)
    kxx = kx**2; kyy = ky**2; kzz = kz**2
    Kxx, Kyy, Kzz = torch.meshgrid(kxx, kyy, kzz, indexing='ij')
    K2 = (Kxx + Kyy + Kzz)  # (S,S,S)

    U   = tfft.fftn(u, dim=(1,2,3))
    S1  = tfft.fftn(u / dt, dim=(1,2,3))
    Nl  = tfft.fftn(u**3,  dim=(1,2,3))
    s_hat = S1 - Nl + 2.0 * K2 * U

    denom = (1.0/dt) + (1.0 - float(eps)) + (K2**2)
    v_hat = s_hat / denom
    up = tfft.ifftn(v_hat, dim=(1,2,3)).real
    return up.unsqueeze(-1)
'''

def _semi_implicit_step_sh(u_in, dt, dx, eps):
    """
    SH3D teacher step (matches MATLAB):
      û^{n+1} = [ (1/dt) û^n - FFT((u^n)^3) + 2 k^2 û^n ] / [ 1/dt + (1-ε) + k^4 ]
    Notes:
      - no dealiasing on u^3 (to match the generator)
      - k^2 = (2π/L)^2 |k|^2 from _k_spectrum(...)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)     # (2π/L)^2 |k|^2
    k4 = k2 * k2

    u0_hat = torch.fft.fftn(u0, dim=(1,2,3))
    u3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))  # no dealias (data)

    denom  = (1.0/dt) + (1.0 - eps) + k4
    numer  = (1.0/dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat

    u1_hat = numer / denom
    u1     = torch.fft.ifftn(u1_hat, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)


###### CH

def pde_rhs_CH(u, dx, eps2):
    """
    Standard CH RHS (continuous PDE):
    u_t = Δ μ,   μ = -eps2 Δ u + (u^3 - u)
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)
    mu    = -eps2 * lap_u + (u**3 - 1.0*u)
    lap_mu = laplacian_fourier_3d_phys(mu, dx)
    return lap_mu

#####
def _get_k2_grid_cached(S, dx, device):
    """
    Cache |k|^2 grid per (S, dx, device) to avoid recomputing every batch.
    """
    key = (S, float(dx), str(device))
    if not hasattr(_get_k2_grid_cached, "_cache"):
        _get_k2_grid_cached._cache = {}
    cache = _get_k2_grid_cached._cache
    if key in cache:
        return cache[key]

    import math, torch
    k = 2.0 * math.pi * torch.fft.fftfreq(S, d=dx).to(device)  # (S,)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = (kx**2 + ky**2 + kz**2)
    cache[key] = k2
    return k2

def _split_hist_coords(x):
    """
    Split x into (hist, coords) assuming the last 3 channels are coordinates
    when present. Falls back to (x, None) if no coords.
    x: (B, S, S, S, C)
    """
    C = x.shape[-1]
    T_in = getattr(config, "T_IN_CHANNELS", 4)
    if C >= T_in + 3:
        hist   = x[..., :T_in]     # time/history channels
        coords = x[..., T_in:]     # 3 coord channels
    else:
        hist, coords = x, None
    return hist, coords
import torch
import torch.nn.functional as F



# ===== Swift–Hohenberg (SH3D) =====

def _r_sh():
    # MATLAB step uses (1 - epsilon) in the implicit linear term → r = 1 - epsilon
    return 1.0 - float(config.EPSILON_PARAM)

def energy_density_SH(u, dx):
    """
    Lyapunov density for SH (gradient flow, unit mobility):
      E(u) = ∫ [ 1/2 |(1 + ∇^2) u|^2  - (r/2) u^2  + (1/4) u^4 ] dx,
    with r = 1 - EPSILON_PARAM.
    """
    r = _r_sh()
    lap_u = laplacian_fourier_3d_phys(u, dx)
    one_plus_lap_u = u + lap_u
    term_lin = 0.5 * (one_plus_lap_u ** 2)
    term_r   = -0.5 * r * (u ** 2)
    term_nl  = 0.25 * (u ** 4)
    return term_lin + term_r + term_nl


def semi_implicit_step_sh(u_in, dt, dx, eps_param):
    """
    One semi-implicit SH step that matches the MATLAB generator:
        s_hat = FFT(u/dt) - FFT(u^3) + 2 k^2 FFT(u)
        v_hat = s_hat / (1/dt + (1 - epsilon) + k^4)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape

    k = 2.0 * math.pi * torch.fft.fftfreq(S, d=dx).to(u0.device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = (kx**2 + ky**2 + kz**2)
    k4 = k2**2

    U0   = torch.fft.fftn(u0, dim=(1,2,3))
    U3   = torch.fft.fftn(u0**3, dim=(1,2,3))
    sHat = U0 / dt - U3 + 2.0 * k2 * U0
    denom = (1.0/dt) + (1.0 - float(eps_param)) + k4
    VHat  = sHat / (denom + 1e-12)

    u1 = torch.fft.ifftn(VHat, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)

#####################################

@torch.no_grad()
def _fft_freqs(S, dx, device):
    k = 2*math.pi*torch.fft.fftfreq(S, d=dx).to(device)  # physical wavenumbers
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = kx*kx + ky*ky + kz*kz
    return k2  # (S,S,S)


def low_k_mse(u_pred, u_ref, frac=0.45):
    """
    MSE between u_pred and u_ref restricted to low spatial frequencies.
    Shapes: (B,S,S,S,1).
    """
    up = u_pred.squeeze(-1).float()
    ur = u_ref.squeeze(-1).float()
    B, S, _, _ = up.shape

    Up = torch.fft.fftn(up, dim=(1,2,3))
    Ur = torch.fft.fftn(ur, dim=(1,2,3))

    fx = torch.fft.fftfreq(S, d=1.0).to(up.device)
    fy = torch.fft.fftfreq(S, d=1.0).to(up.device)
    fz = torch.fft.fftfreq(S, d=1.0).to(up.device)
    FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    rmax = r.max()
    mask = (r <= frac * rmax).to(Up.real.dtype)

    Dh = Up - Ur
    spec_mse = (Dh.real**2 + Dh.imag**2) * mask
    # normalize by # of active modes to keep scale stable across S
    denom = mask.sum().clamp_min(1.0)
    return spec_mse.sum() / denom
def low_k_mse_T_out(u_pred, u_ref, frac=0.45):
    """
    MSE between u_pred and u_ref restricted to low spatial frequencies.

    Supports shapes:
      - (B,S,S,S)          single-step
      - (B,S,S,S,1)        single-step with channel
      - (B,S,S,S,T_out)    multi-step (time/channel last)

    For multi-step, we treat each time slice as an extra batch element
    (so the loss aggregates over all steps).
    """
    up = u_pred
    ur = u_ref

    # --- ensure 5D with time/channel last ---
    if up.dim() == 4:
        up = up.unsqueeze(-1)   # (B,S,S,S,1)
    if ur.dim() == 4:
        ur = ur.unsqueeze(-1)

    assert up.dim() == 5 and ur.dim() == 5, "low_k_mse expects 4D or 5D inputs"

    # --- align time dimension ---
    T_p = up.shape[-1]
    T_r = ur.shape[-1]

    if T_p != T_r:
        # broadcast the one-step teacher across time if needed
        if T_p > 1 and T_r == 1:
            ur = ur.expand(*ur.shape[:-1], T_p)
            T_r = T_p
        elif T_r > 1 and T_p == 1:
            up = up.expand(*up.shape[:-1], T_r)
            T_p = T_r
        else:
            raise ValueError(f"low_k_mse: incompatible time dims {T_p} vs {T_r}")

    # now up, ur: (B,S,S,S,T) with same T
    B, S, _, _, T = up.shape

    # flatten time into batch: (B*T,S,S,S,1)
    up_flat = up.permute(0, 4, 1, 2, 3).reshape(B*T, S, S, S, 1)
    ur_flat = ur.permute(0, 4, 1, 2, 3).reshape(B*T, S, S, S, 1)

    up4 = up_flat.squeeze(-1).float()  # (B*T,S,S,S)
    ur4 = ur_flat.squeeze(-1).float()

    # --- original spectral low-k MSE on the flattened batch ---
    Up = torch.fft.fftn(up4, dim=(1, 2, 3))
    Ur = torch.fft.fftn(ur4, dim=(1, 2, 3))

    fx = torch.fft.fftfreq(S, d=1.0).to(up4.device)
    fy = torch.fft.fftfreq(S, d=1.0).to(up4.device)
    fz = torch.fft.fftfreq(S, d=1.0).to(up4.device)
    FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    rmax = r.max()
    mask = (r <= frac * rmax).to(Up.real.dtype)

    Dh = Up - Ur
    spec_mse = (Dh.real**2 + Dh.imag**2) * mask

    # same normalization style as before: by # of active modes only
    denom = mask.sum().clamp_min(1.0)
    return spec_mse.sum() / denom

# --- CH projection with mixed chemistry (uses the same MATLAB-matched kernel) ---

def _k_vectors(nx, ny, nz, dx, device, dtype=torch.float32):
    kx = torch.fft.fftfreq(nx, d=dx).to(device)
    ky = torch.fft.fftfreq(ny, d=dx).to(device)
    kz = torch.fft.fftfreq(nz, d=dx).to(device)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    # no in-place ops
    kx = kx * (2*np.pi); ky = ky * (2*np.pi); kz = kz * (2*np.pi)
    return kx.to(dtype), ky.to(dtype), kz.to(dtype)

# --- H^{-1} distance between two scalar fields (B,S,S,S) ---
def hminus1_mse(u, v, dx, eps_floor_scale=1e-2):
    """
    Return mean_{batch,k} |Û-Ṽ|^2 / (k^2 + eps0), i.e., an H^{-1} metric.
    This emphasizes low-k agreement (CH gradient flow is H^{-1}).
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, S, _, _ = u.shape
        # safe inverse k^2 (reuse your k-spectrum convention)
        kx = torch.fft.fftfreq(S, d=dx).to(u.device)
        ky = torch.fft.fftfreq(S, d=dx).to(u.device)
        kz = torch.fft.fftfreq(S, d=dx).to(u.device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        k2 = (2*np.pi)**2 * (KX**2 + KY**2 + KZ**2)
        # floor ~ small fraction of (2π/L)^2
        L = S * dx
        eps0 = (2*np.pi/L)**2 * eps_floor_scale
        inv_k2 = 1.0 / (k2 + eps0)

        d  = (u.float() - v.float())
        dh = torch.fft.fftn(d, dim=(1,2,3))
        pow_spec = dh.real**2 + dh.imag**2
        val = (pow_spec * inv_k2).mean()
        return val.to(u.dtype)


# --- Simple spectral low-pass (keep lowest frac of modes) ---
def lowpass_field(u, frac=0.35):
    """
    Keep modes with radius <= frac*rmax (index space). Good enough to
    isolate the coarse μ that drives CH coarsening.
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, S, _, _ = u.shape
        fx = torch.fft.fftfreq(S, d=1.0).to(u.device)
        fy = torch.fft.fftfreq(S, d=1.0).to(u.device)
        fz = torch.fft.fftfreq(S, d=1.0).to(u.device)
        FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing='ij')
        r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
        rmax = r.max()
        mask = (r <= frac * rmax).float()

        uh = torch.fft.fftn(u.float(), dim=(1,2,3))
        ul = torch.fft.ifftn(uh * mask, dim=(1,2,3)).real
        return ul.to(u.dtype)


def interface_weight(u, dx, eps=1e-12):
    """
    Weight in [0,1] that highlights interfaces.
    We use normalized |∇u| as a proxy for interface location.
    """
    ux, uy, uz = grad_fourier(u, dx)     # spectral gradient (cheap & alias-free)
    g = torch.sqrt(ux*ux + uy*uy + uz*uz + eps)
    gmax = g.amax(dim=(1,2,3), keepdim=True) + 1e-12
    return (g / gmax).clamp(0, 1)



def laplacian_fourier_3d_phys(u, dx):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, nx, ny, nz = u.shape
        minus_k2 = _fft_wavenumbers_3d(nx, ny, nz, dx).to(u.device).to(torch.float32)
        u32 = u.float()
        u_ft = torch.fft.fftn(u32, dim=[1,2,3])
        lap  = torch.fft.ifftn(minus_k2 * u_ft, dim=[1,2,3]).real
    return lap.to(u.dtype)

def biharmonic(u, dx):
    # ∇⁴u = ∇²(∇²u)
    return laplacian_fourier_3d_phys(laplacian_fourier_3d_phys(u, dx), dx)

def triharmonic(u, dx):
    # ∇⁶u = ∇²(∇⁴u)
    return laplacian_fourier_3d_phys(biharmonic(u, dx), dx)

def grad_fourier(u, dx):
    """
    Spectral gradient: returns (ux, uy, uz) same dtype/device as u
    """
    B, nx, ny, nz = u.shape
    kx, ky, kz = _k_vectors(nx, ny, nz, dx, u.device, dtype=torch.float32)
    uhat = torch.fft.fftn(u.float(), dim=[1,2,3])
    i = torch.complex(torch.tensor(0.0, device=u.device), torch.tensor(1.0, device=u.device))
    ux = torch.fft.ifftn(i * kx * uhat, dim=[1,2,3]).real
    uy = torch.fft.ifftn(i * ky * uhat, dim=[1,2,3]).real
    uz = torch.fft.ifftn(i * kz * uhat, dim=[1,2,3]).real
    return ux.to(u.dtype), uy.to(u.dtype), uz.to(u.dtype)
def div_fourier(vx, vy, vz, dx):
    """
    Spectral divergence of vector field v = (vx, vy, vz)
    """
    B, nx, ny, nz = vx.shape
    kx, ky, kz = _k_vectors(nx, ny, nz, dx, vx.device, dtype=torch.float32)
    i = torch.complex(torch.tensor(0.0, device=vx.device), torch.tensor(1.0, device=vx.device))
    vxhat = torch.fft.fftn(vx.float(), dim=[1,2,3])
    vyhat = torch.fft.fftn(vy.float(), dim=[1,2,3])
    vzhat = torch.fft.fftn(vz.float(), dim=[1,2,3])
    div_hat = i * (kx * vxhat + ky * vyhat + kz * vzhat)
    return torch.fft.ifftn(div_hat, dim=[1,2,3]).real.to(vx.dtype)

# -------------------------
# Dealiasing (kept)
# -------------------------
def dealias_two_thirds(u):
    S = u.shape[1]
    kcut = S // 3
    filt = torch.zeros((S, S, S//2 + 1), device=u.device, dtype=torch.float32)
    filt[:2*kcut, :2*kcut, :kcut+1] = 1.0
    uhat = torch.fft.rfftn(u, dim=(1,2,3))
    return torch.fft.irfftn(uhat * filt, s=(S,S,S), dim=(1,2,3)).real

# -------------------------
# RHS dispatch per PROBLEM

def _rhs_ac3d(u, dx, eps2):
    # Dataset-consistent: NO dealiasing in the cubic (matches MATLAB generator)
    lap_u = laplacian_fourier_3d_phys(u, dx)
    return lap_u - (1.0/eps2) * (u**3 - u)


def _rhs_mbe3d(u, dx, eps):
    """
    MBE3D (slope-selection, data-consistent):
        u_t = -Δ u  -  eps ∇^4 u  +  ∇·(|∇u|^2 ∇u)
             = -ε Δ^2 u - ∇·((1 - |∇u|^2)∇u)
    """
    ux, uy, uz = grad_fourier(u, dx)
    s = ux*ux + uy*uy + uz*uz
    vx, vy, vz = s*ux, s*uy, s*uz
    div_f = div_fourier(vx, vy, vz, dx)    # ∇·(|∇u|^2 ∇u)
    lap   = laplacian_fourier_3d_phys(u, dx)
    bi    = biharmonic(u, dx)
    return -lap - eps * bi + div_f



def _rhs_pfc3d(u, dx, eps):
    # u_t = (1-ε)∇²u + 2∇⁴u + ∇⁶u - ∇²(u^3)
    lap_u = laplacian_fourier_3d_phys(u, dx)
    bi_u  = biharmonic(u, dx)
    tri_u = triharmonic(u, dx)
    # match data: NO dealiasing in the cubic
    lap_u3 = laplacian_fourier_3d_phys(u**3, dx)
    return (1.0 - eps) * lap_u + 2.0 * bi_u + tri_u - lap_u3

def _rhs_sh3d(u, dx, eps):
    """
    Swift–Hohenberg (data-consistent split):
        u_t = (1-ε) u - 2 Δ u - Δ^2 u - u^3
    NOTE:
      • NO dealiasing on u^3 (matches the MATLAB generator).
      • Clamp inside the cubic to avoid overflow when the net overshoots early.
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)     # Δu
    bi_u  = biharmonic(u, dx)                    # Δ^2 u
    u_safe = torch.clamp(u, -5.0, 5.0)           # safe cubic; no in-place to keep graph
    return (1.0 - eps) * u - 2.0 * lap_u - bi_u - (u_safe ** 3)

def semi_implicit_step_pfc(u_in, dt, dx, eps):
    """
    One PFC3D semi-implicit step (matches your MATLAB generator).
    Inputs:  u_in (B,S,S,S,1), dt, dx, eps
    Returns: (B,S,S,S,1)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)^2 |k|^2  >= 0

    U0     = torch.fft.fftn(u0,    dim=(1,2,3))
    U3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))  # no dealias (data)

    numer = (1.0/dt) * U0 - k2 * U3_hat + 2.0 * (k2**2) * U0
    denom = (1.0/dt) + (1.0 - eps) * k2 + (k2**3)

    U1 = numer / denom
    u1 = torch.fft.ifftn(U1, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)


def _scheme_residual_fourier_pfc(u0, up, dt, dx, eps):
    # u0, up: (B,S,S,S)
    k2 = _k_spectrum(u0.shape[1], u0.shape[2], u0.shape[3], dx, u0.device)
    U0     = torch.fft.fftn(u0,    dim=(1,2,3))
    UP     = torch.fft.fftn(up,    dim=(1,2,3))
    U3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))    # data-consistent (no dealias)

    denom  = (1.0/dt) + (1.0 - eps)*k2 + (k2**3)
    rhs    = (1.0/dt)*U0 - k2*U3_hat + 2.0*(k2**2)*U0
    rhat   = denom*UP - rhs
    rhat   = rhat / (denom + 1e-12)                # precondition like SH
    return (rhat.real**2 + rhat.imag**2).mean()


def pde_rhs(u, dx, eps_param):
    """
    Generic RHS(u) for the active problem such that u_t = RHS(u).
    """
    P = config.PROBLEM
    if P == 'AC3D':
        return _rhs_ac3d(u, dx, config.EPS2)
    elif P == 'CH3D':
        return _rhs_ch3d(u, dx, eps_param)
    elif P == 'SH3D':
        return _rhs_sh3d(u, dx, eps_param)
    elif P == 'MBE3D':
        return _rhs_mbe3d(u, dx, eps_param)
    elif P == 'PFC3D':
        return _rhs_pfc3d(u, dx, eps_param)
    else:
        raise ValueError(f"Unknown PROBLEM '{P}'")

# -------------------------
# Allen–Cahn chemical potential alias (compat)
# For non-AC problems, we return RHS(u) so your debug prints still work.
# -------------------------
def mu_ac(u, dx, eps2, dealias=True):
    if config.PROBLEM == 'AC3D':
        lap_u = laplacian_fourier_3d_phys(u, dx)
        if dealias:
            u = dealias_two_thirds(u)
        return lap_u - (1.0/eps2) * (u**3 - u)
    # For other PDEs, use RHS(u) as a general “μ-like” term for logging
    return pde_rhs(u, dx, config.EPSILON_PARAM)

# -------------------------
# Physics residuals (now generic)
# -------------------------
def physics_residual_matlab(u_in, u_pred):
    # Preserved for backward compat (Allen–Cahn form). Kept but now generic.
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    ut = (up - u0) / dt
    mu = pde_rhs(up, dx, config.EPSILON_PARAM)
    R = ut - mu
    mse_phys = F.mse_loss(R, torch.zeros_like(R))
    debug_ut_mse = torch.mean(ut**2)
    debug_muspatial_mse = config.DEBUG_MU_SCALE * torch.mean(mu**2)
    return mse_phys, debug_ut_mse, debug_muspatial_mse

def physics_residual_normalized(u_in, u_pred):
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    ut = (up - u0) / dt
    mu = pde_rhs(up, dx, config.EPSILON_PARAM)
    s_t  = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    s_mu = mu.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    R_tilde = ut / s_t - mu / s_mu
    loss = R_tilde.pow(2).mean()
    return loss, s_t.mean(), s_mu.mean()

def physics_residual_midpoint(u_in, u_pred):
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    um = 0.5 * (u0 + up)
    ut = (up - u0) / dt
    mu_m = pde_rhs(um, dx, config.EPSILON_PARAM)
    Rm = ut - mu_m
    return (Rm**2).mean()

# -------------------------
# Semi-implicit / teacher step
# -------------------------
def semi_implicit_step(u_in, dt, dx, eps2):
    """
    AC3D: keep your previous semi-implicit.
    Others: safe explicit Euler teacher (one-step) using RHS(u^n).
    """
    P = config.PROBLEM
    if P == 'AC3D':
        # original AC3D semi-implicit step
        u0 = u_in.squeeze(-1).float()
        B,S,_,_ = u0.shape[:4]
        kx = torch.fft.fftfreq(S, d=dx).to(u0.device)
        ky = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kz = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k2 = (2*np.pi)**2 * (kx**2 + ky**2 + kz**2)
        nl = u0**3 - u0
        u0_hat = torch.fft.fftn(u0, dim=[1,2,3])
        nl_hat = torch.fft.fftn(nl, dim=[1,2,3])
        num = u0_hat - (dt/eps2) * nl_hat
        den = (1.0 + dt * k2)
        u1_hat = num / den
        u1 = torch.fft.ifftn(u1_hat, dim=[1,2,3]).real

        return u1.unsqueeze(-1)

    else:
        # fallback (unchanged) explicit Euler
        u0 = u_in.squeeze(-1)
        rhs = pde_rhs(u0, dx, config.EPSILON_PARAM)
        return (u0 + dt * rhs).unsqueeze(-1)

def semi_implicit_step_T_out(u_in, dt, dx, eps2):
    """
    AC3D: semi-implicit scheme.
    - If T_OUT == 1: behaves as before, returns one step u^{n+1}.
    - If T_OUT > 1 : returns [u^{n+1}, ..., u^{n+T_OUT}] in the last dim.

    Others: explicit Euler teacher (also extended to T_OUT steps).
    """
    P = config.PROBLEM
    T_out = int(getattr(config, "T_OUT", 1))

    # last known state u^n: (B,S,S,S)
    u0 = u_in.squeeze(-1).float()

    if P == 'AC3D':
        # original AC3D semi-implicit, extended to multiple steps
        B, S, _, _ = u0.shape[:4]

        # spectral grid (same for all steps)
        kx = torch.fft.fftfreq(S, d=dx).to(u0.device)
        ky = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kz = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k2 = (2 * np.pi) ** 2 * (kx**2 + ky**2 + kz**2)
        den = (1.0 + dt * k2)

        u_cur = u0
        steps = []
        for _ in range(T_out):
            nl = u_cur**3 - u_cur
            u0_hat = torch.fft.fftn(u_cur, dim=[1, 2, 3])
            nl_hat = torch.fft.fftn(nl, dim=[1, 2, 3])
            num = u0_hat - (dt / eps2) * nl_hat
            u1_hat = num / den
            u_next = torch.fft.ifftn(u1_hat, dim=[1, 2, 3]).real  # (B,S,S,S)

            steps.append(u_next)
            u_cur = u_next

        u_all = torch.stack(steps, dim=-1)  # (B,S,S,S,T_out)
        return u_all  # for T_out=1, shape is (B,S,S,S,1)

    else:
        # fallback explicit Euler, extended to multiple steps
        u_cur = u0
        steps = []
        for _ in range(T_out):
            rhs = pde_rhs(u_cur, dx, config.EPSILON_PARAM)
            u_next = u_cur + dt * rhs
            steps.append(u_next)
            u_cur = u_next

        u_all = torch.stack(steps, dim=-1)  # (B,S,S,S,T_out)
        return u_all

# -------------------------
# Energy utilities
# For AC3D use your original. Otherwise return zero penalty (no-op).
# -------------------------
def energy_density(u, dx, eps2):
    lap = laplacian_fourier_3d_phys(u, dx)
    grad2_term = -0.5 * eps2 * (u * lap)
    pot_term   = 0.25 * (u**2 - 1.0)**2
    return grad2_term + pot_term

def energy_penalty(u_in, u_pred, dx, eps2):
    # CH is also a (mass-conserving) gradient flow of the same free energy.
    if config.PROBLEM not in ('AC3D', 'CH3D'):
        return torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    E0 = energy_density(u0, dx, eps2).mean(dim=(1,2,3))
    Ep = energy_density(up, dx, eps2).mean(dim=(1,2,3))
    inc = torch.relu(Ep - E0)
    return inc.mean()


''''
def energy_penalty(u_ref, u_pred, dx, eps2):
    """
    Match free energy of u_pred to u_ref at each time step.

    u_ref, u_pred shapes:
        - (B,S,S,S,T)  multi-step
        - or (B,S,S,S) single-step (then treated as T=1)
    """
    if config.PROBLEM not in ('AC3D', 'CH3D'):
        return torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    ur = u_ref
    up = u_pred

    # unify shapes
    if ur.dim() == 5 and ur.shape[-1] == 1:
        ur = ur.squeeze(-1)
    if up.dim() == 5 and up.shape[-1] == 1:
        up = up.squeeze(-1)

    # single-step: just L2 on energy
    if ur.dim() == 4 and up.dim() == 4:
        E_ref = energy_density(ur, dx, eps2).mean(dim=(1, 2, 3))  # (B,)
        E_pred = energy_density(up, dx, eps2).mean(dim=(1, 2, 3)) # (B,)
        return ((E_pred - E_ref)**2).mean()

    # multi-step: (B,S,S,S,T)
    if ur.dim() == 5 and up.dim() == 5:
        assert ur.shape == up.shape
        B, Sx, Sy, Sz, T = ur.shape

        ur_flat = ur.permute(0, 4, 1, 2, 3).reshape(B*T, Sx, Sy, Sz)
        up_flat = up.permute(0, 4, 1, 2, 3).reshape(B*T, Sx, Sy, Sz)

        E_ref_all = energy_density(ur_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)
        E_pred_all = energy_density(up_flat, dx, eps2).mean(dim=(1, 2, 3)) # (B*T,)

        return ((E_pred_all - E_ref_all)**2).mean()

    # fallback
    E_ref = energy_density(ur, dx, eps2).mean(dim=tuple(range(1, ur.dim())))
    E_pred = energy_density(up, dx, eps2).mean(dim=tuple(range(1, up.dim())))
    return ((E_pred - E_ref)**2).mean()
'''

def energy_penalty_T_out(u_ref, u_pred, dx, eps2):
    """
    Match free energy of u_pred to u_ref.

    Supported shapes:
      - u_ref, u_pred: (B,S,S,S)                 single-step
      - u_ref, u_pred: (B,S,S,S,T)               multi-step
      - u_ref: (B,S,S,S),    u_pred: (B,S,S,S,T) baseline vs multi-step
      - also accepts trailing singleton time dims (B,S,S,S,1).
    """
    if config.PROBLEM not in ('AC3D', 'CH3D'):
        return torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    ur = u_ref
    up = u_pred

    # strip useless trailing singleton time dims
    if ur.dim() == 5 and ur.shape[-1] == 1:
        ur = ur.squeeze(-1)
    if up.dim() == 5 and up.shape[-1] == 1:
        up = up.squeeze(-1)

    # ---- case 1: both single-step (B,S,S,S) ----
    if ur.dim() == 4 and up.dim() == 4:
        E_ref  = energy_density(ur, dx, eps2).mean(dim=(1, 2, 3))  # (B,)
        E_pred = energy_density(up, dx, eps2).mean(dim=(1, 2, 3))  # (B,)
        return ((E_pred - E_ref) ** 2).mean()

    # ---- case 2: both multi-step (B,S,S,S,T) ----
    if ur.dim() == 5 and up.dim() == 5:
        assert ur.shape == up.shape
        B, Sx, Sy, Sz, T = ur.shape

        ur_flat = ur.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
        up_flat = up.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)

        E_ref_all  = energy_density(ur_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)
        E_pred_all = energy_density(up_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)

        return ((E_pred_all - E_ref_all) ** 2).mean()

    # ---- case 3: ref single-step, pred multi-step ----
    if ur.dim() == 4 and up.dim() == 5:
        B, Sx, Sy, Sz, T = up.shape

        # broadcast u_ref over time steps
        ur_flat = ur.unsqueeze(1).expand(B, T, Sx, Sy, Sz).reshape(B * T, Sx, Sy, Sz)
        up_flat = up.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)

        E_ref_all  = energy_density(ur_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)
        E_pred_all = energy_density(up_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)

        return ((E_pred_all - E_ref_all) ** 2).mean()

    # ---- case 4: ref multi-step, pred single-step (rare) ----
    if ur.dim() == 5 and up.dim() == 4:
        # just swap roles → reuse case 3
        return energy_penalty(up, ur, dx, eps2)

    # ---- fallback: very unusual shapes, just compare global energies ----
    E_ref  = energy_density(ur, dx, eps2).mean(dim=tuple(range(1, ur.dim())))
    E_pred = energy_density(up, dx, eps2).mean(dim=tuple(range(1, up.dim())))
    return ((E_pred - E_ref) ** 2).mean()


def mass_penalty(u_in, u_pred):
    """Penalize change in spatial mean (mass) between steps."""
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    m0 = u0.mean(dim=(1,2,3))
    mp = up.mean(dim=(1,2,3))
    return ((mp - m0)**2).mean()

def mass_project_pred(y_pred, u_in_last):
    """Hard projection: force predicted mean to equal input mean (differentiable)."""
    m_in  = u_in_last.mean(dim=(1,2,3,4), keepdim=True)
    m_out = y_pred.mean(dim=(1,2,3,4),   keepdim=True)
    return y_pred - m_out + m_in


# -------------------------
# Minimizing-movement projection (generic: use RHS)
# -------------------------
def mm_projection(u_in, u_pred, dt, dx, eps2, steps=5, eta=None):
    up = u_pred.squeeze(-1).float()
    u0 = u_in.squeeze(-1).float()
    if eta is None:
        eta = 0.5 * dt
    for _ in range(steps):
        g = (up - u0) / dt - pde_rhs(up, dx, config.EPSILON_PARAM).float()
        up = up - eta * g
    return up.unsqueeze(-1), g

def loss_mm_projection(u_in, u_pred):
    dt, dx = config.DT, config.DX
    up_ref, g_last = mm_projection(u_in, u_pred, dt, dx, config.EPS2, steps=5)
    l_proj = F.mse_loss(u_pred, up_ref)
    l_stat = (g_last**2).mean()
    return l_proj + l_stat, l_proj.detach(), l_stat.detach()

# -------------------------
# Scheme residual (teacher consistency)
# Generic: compare to explicit Euler from u^n.
# -------------------------
def scheme_residual_fourier(u_in, u_pred):
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    if config.PROBLEM == 'CH3D':
        # ----- EXACT MATLAB SEMI-IMPLICIT RESIDUAL (minimal change) -----
        B, S, _, _ = u0.shape
        k2 = _k_spectrum(S, S, S, dx, u0.device)       # (2π/L)^2 |k|^2  >= 0
        k4 = k2 * k2

        u0_hat    = torch.fft.fftn(u0, dim=(1,2,3))
        up_hat    = torch.fft.fftn(up, dim=(1,2,3))
        chem0_hat = torch.fft.fftn(u0**3 - 3.0*u0, dim=(1,2,3))

        eps = config.EPSILON_PARAM
        denom = 1.0 + dt * (2.0 * k2 + (eps * eps) * k4)            # 1 + dt(2k^2 + eps^2 k^4)

        # R^ = denom * û^{n+1} - [ û^n - dt * k^2 * (u^3 - 3u)^n^ ]
        rhat = denom * up_hat - (u0_hat - dt * k2 * chem0_hat)

        # Ignore k=0 mass mode (nullspace) exactly as in CH
        rhat = rhat * (k2 > 0)

        # Precondition by the same denom to avoid high-k domination
        rhat = rhat / (denom + 1e-12)

        # Parseval: ||R||^2 ~ sum |R^|^2. Use mean to keep scale stable.
        r2 = rhat.real**2 + rhat.imag**2
        return r2.mean()
    if config.PROBLEM == 'SH3D':  # <-- NEW
        B, S, _, _ = u0.shape
        k2 = _k_spectrum(S, S, S, dx, u0.device)
        k4 = k2 * k2

        u0_hat = torch.fft.fftn(u0, dim=(1, 2, 3))
        up_hat = torch.fft.fftn(up, dim=(1, 2, 3))
        u3_hat = torch.fft.fftn(u0 ** 3, dim=(1, 2, 3))  # data-consistent (no dealias)

        eps = config.EPSILON_PARAM
        denom = (1.0 / dt) + (1.0 - eps) + k4

        # residual in Fourier of the semi-implicit update
        rhs_hat = (1.0 / dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat
        rhat = denom * up_hat - rhs_hat

        # precondition to avoid high-k domination (same idea as CH)
        rhat = rhat / (denom + 1e-12)

        r2 = rhat.real ** 2 + rhat.imag ** 2
        return r2.mean()

    elif config.PROBLEM == 'PFC3D':  # NEW
        return _scheme_residual_fourier_pfc(u0, up, dt, dx, config.EPSILON_PARAM)

    elif config.PROBLEM == 'MBE3D':
        # Semi-implicit MBE residual (preconditioned in Fourier)
        B, S, _, _ = u0.shape
        k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)^2 |k|^2
        k4 = k2 * k2

        # denom = 1/dt - k^2 + eps * k^4   (matches MATLAB: 1/dt - (pp2+qq2+rr2) + eps*(...)^2 )
        eps = config.EPSILON_PARAM
        denom = (1.0 / dt) - k2 + eps * k4

        # rhs_hat = FFT( u0/dt + div( |∇u0|^2 ∇u0 ) )
        ux, uy, uz = grad_fourier(u0, dx)  # ∇u0 in real space
        s = ux * ux + uy * uy + uz * uz
        vx, vy, vz = s * ux, s * uy, s * uz
        div_term = div_fourier(vx, vy, vz, dx)  # real space
        rhs_real = (1.0 / dt) * u0 + div_term

        U0p = torch.fft.fftn(up, dim=(1, 2, 3))
        RHS = torch.fft.fftn(rhs_real, dim=(1, 2, 3))

        rhat = denom * U0p - RHS
        # precondition to avoid high-k domination (same style as other PDEs)
        rhat = rhat / (denom + 1e-12)

        r2 = rhat.real ** 2 + rhat.imag ** 2
        return r2.mean()



    if config.PROBLEM == 'AC3D':
        u_explicit = u0 + dt * pde_rhs(u0, dx, config.EPSILON_PARAM).float()
        r = up - u_explicit
        return (r**2).mean()


##
##
# ---------- Fourier grid ----------
def _kgrid(nx, dx, device):
    # like MATLAB: [0:N/2, -N/2+1:-1]
    k = 2.0*torch.pi/(nx*dx) * torch.cat([
        torch.arange(0, nx//2 + 1, device=device),
        torch.arange(-nx//2 + 1, 0,        device=device)
    ])
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k4 = k2**2
    return k2, k4

# ---------- CH semi-implicit residual from your MATLAB step ----------
@torch.no_grad()
def _safe_den(den):
    return den + 1e-12

def ch3d_semiimplicit_residual(u_n, u_np1, dt, dx, eps, huber_beta=1.0):
    """
    u_n, u_np1: (B,S,S,S) in real space
    returns scalar physics loss for CH (semi-implicit residual, robust & preconditioned)
    """
    _, S, _, _ = u_n.shape
    device = u_n.device
    k2, k4 = _kgrid(S, dx, device)
    eps2 = eps**2

    u0_hat   = torch.fft.fftn(u_n,   dim=(1,2,3))
    up_hat   = torch.fft.fftn(u_np1, dim=(1,2,3))
    chem0    = u_n**3 - 3.0*u_n                 # f'(u) = u^3 - 3u  (your code)
    chem0_hat= torch.fft.fftn(chem0, dim=(1,2,3))

    denom = 1.0 + dt*(2.0*k2 + eps2*k4)         # semi-implicit operator
    lhs   = denom * up_hat
    rhs   = u0_hat - dt * k2 * chem0_hat
    rhat  = lhs - rhs

    # CH specifics:
    mask = (k2 > 0).to(rhat.real.dtype)         # ignore k=0 mass mode (nullspace)
    rhat = rhat * mask

    # precondition: whiten spectrum so high-k doesn’t dominate
    rhat = rhat / _safe_den(denom)

    # robust penalty in Fourier (Parseval)
    r2 = rhat.real**2 + rhat.imag**2
    loss_res = F.smooth_l1_loss(r2, torch.zeros_like(r2), beta=huber_beta)
    return loss_res

# ---------- tiny soft anchor for mass (k=0 mode) ----------
def ch_mass_anchor(u_n, u_np1):
    m0 = u_n.mean(dim=(1,2,3))
    m1 = u_np1.mean(dim=(1,2,3))
    return ((m1 - m0)**2).mean()

# ---------- one-sided (hinge) free-energy decrease ----------
def _periodic_grad2(u, dx):
    # periodic forward differences; avoids FFT normalization pitfalls
    gx = (torch.roll(u, shifts=-1, dims=1) - u) / dx
    gy = (torch.roll(u, shifts=-1, dims=2) - u) / dx
    gz = (torch.roll(u, shifts=-1, dims=3) - u) / dx
    return gx*gx + gy*gy + gz*gz

def ch_free_energy_density(u, dx, eps):
    bulk = 0.25*(u**2 - 1.0)**2
    grad2 = _periodic_grad2(u, dx)
    return (bulk + 0.5*(eps**2)*grad2).mean()

def ch_energy_hinge(u_n, u_np1, dx, eps):
    Fn  = ch_free_energy_density(u_n,   dx, eps)
    Fnp = ch_free_energy_density(u_np1, dx, eps)
    return torch.relu(Fnp - Fn)    # penalize only increases


##

def _mid_residual_norm_sh(u_in, u_pred):
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    um = 0.5 * (u0 + up)
    ut = (up - u0) / dt
    lap = laplacian_fourier_3d_phys(um, dx)
    bi = biharmonic(um, dx)
    rhs = (1.0 - eps) * um - 2.0 * lap - bi - (um ** 3)
    s_t = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
    s_r = rhs.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
    R = ut / s_t - rhs / s_r
    return (R ** 2).mean()


##
##############



def _hm1_mean(R, dx, eps_floor_scale=1e-2):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, S, _, _ = R.shape
        k2, inv_k2 = _fft_rk2(S, S, S, dx, R.device, dtype=torch.float32)
        R_hat = torch.fft.fftn(R.float(), dim=(1,2,3))
        pow_spec = R_hat.real**2 + R_hat.imag**2
        return (pow_spec * inv_k2).mean().to(R.dtype)

# === NEW: one extra Gauss–Lobatto collocation residual in H^{-1} (CH3D) ===
import math


def physics_collocation_tau_L2_SH(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                  normalize=True):
    """
    SH3D collocation residual at interior time u_tau:
      R_tau = (u^{n+1}-u^n)/dt - RHS_SH(u_tau)
    with
      RHS_SH(u) = (1-ε) u - 2 Δ u - Δ^2 u - u^3
    Notes:
      • NO dealiasing on u^3 (matches your MATLAB generator).
      • Optional per-sample normalization balances ut vs RHS in L2.
      • Scored in L2 (SH is an L2-type gradient flow).
    """
    assert config.PROBLEM == 'SH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float()

    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    # Dataset-consistent SH RHS (no dealias on cubic)
    lap_u   = laplacian_fourier_3d_phys(u_tau, dx)
    bi_u    = biharmonic(u_tau, dx)
    rhs_tau = (1.0 - eps) * u_tau - 2.0 * lap_u - bi_u - (u_tau ** 3)

    if normalize:
        s_t = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R ** 2).mean().to(u_pred.dtype)

def sh_free_energy_density(u, dx, eps):
    """
    Per-voxel energy density for Swift–Hohenberg:
      f_SH(u) = - (1-ε)/2 * u^2  - |∇u|^2  + 1/2 * (Δu)^2  + 1/4 * u^4
    Uses periodic finite differences for |∇u|^2 and spectral Δ for Δu.
    Shapes: u (B,S,S,S), returns (B,S,S,S).
    """
    # |∇u|^2 with periodic forward diffs (same style as CH helper)
    grad2 = _periodic_grad2(u, dx)  # already in your file

    # Δu via your spectral Laplacian
    lap = laplacian_fourier_3d_phys(u, dx)

    bulk_quad  = -0.5 * (1.0 - eps) * (u * u)
    grad_term  = -grad2
    bih_term   = 0.5 * (lap * lap)
    quartic    = 0.25 * (u * u * u * u)

    return (bulk_quad + grad_term + bih_term + quartic)

def energy_penalty_sh(u_in, u_pred, dx, eps):
    """
    One-sided hinge on F_SH increase: penalize only if F(u^{n+1}) > F(u^n).
    Returns a scalar (mean over batch).
    """
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    F0 = sh_free_energy_density(u0, dx, eps).mean(dim=(1,2,3))
    Fp = sh_free_energy_density(up, dx, eps).mean(dim=(1,2,3))
    inc = torch.relu(Fp - F0)  # only increases are penalized
    return inc.mean()



# === Optimal Physics-Guided Update for CH3D (minimizes SI residual in H^{-1}) ===
import torch, math
import torch.nn.functional as F
import numpy as np
import config

import torch, math
import torch.nn.functional as F


#############
def physics_collocation_tau_L2_PFC(u_in, u_pred, tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                   normalize=True):
    """
    PFC3D collocation at u_tau: R_tau = (u^{n+1}-u^n)/dt - RHS_PFC(u_tau).
    Scored in L2; optional per-sample normalization.
    """
    assert config.PROBLEM == 'PFC3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    ut   = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up
    # RHS that matches data (no dealias):
    rhs_tau = _rhs_pfc3d(u_tau, dx, eps)

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau
    return (R**2).mean().to(u_pred.dtype)




def pfc_free_energy_density(u, dx, eps):
    """
    F[u] = ∫ [ 1/2(1-ε)u^2  - |∇u|^2  + 1/2(Δu)^2  - 1/4 u^4 ] dx
    δF/δu = (1-ε)u + 2Δu + Δ^2 u - u^3  ⇒  u_t = Δ(δF/δu) matches your PFC step.
    """
    grad2 = _periodic_grad2(u, dx)                      # |∇u|^2
    lap   = laplacian_fourier_3d_phys(u, dx)            # Δu

    bulk_quad = 0.5 * (1.0 - eps) * (u * u)
    grad_term = -grad2
    bih_term  = 0.5 * (lap * lap)
    quartic   = -0.25 * (u ** 4)

    return bulk_quad + grad_term + bih_term + quartic

def energy_penalty_pfc(u_in, u_pred, dx, eps):
    """One-sided hinge: penalize F(u^{n+1}) > F(u^n)."""
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    F0 = pfc_free_energy_density(u0, dx, eps).mean(dim=(1,2,3))
    Fp = pfc_free_energy_density(up, dx, eps).mean(dim=(1,2,3))
    return torch.relu(Fp - F0).mean()


###########
## MBE3D

# ==== MBE3D helpers (append to functions.py) =================================
import math as _math

def _k_axes_phys(S, dx, device, dtype):
    # physical k like MATLAB: 2π/L * [0..S/2, -S/2+1..-1]
    L = S * dx
    half = S // 2
    kv = torch.cat([torch.arange(0, half + 1, device=device),
                    torch.arange(-half + 1, 0, device=device)], dim=0).to(dtype)
    return (2.0 * _math.pi / L) * kv  # (S,)



# --- at top of functions.py (near imports) ---
_MBES2_MAX = 36.0  # gentle cap for |∇u|^2 (tunable: 16..64 works well)

def _cap_s2(s2, cap=_MBES2_MAX):
    # inplace-free, differentiable cap
    return torch.clamp(s2, max=cap)

def semi_implicit_step_mbe(u_in, dt, dx, eps):
    """
    Robust MBE3D semi-implicit, computed in float64 to avoid overflow/underflow.
    """
    u = u_in.squeeze(-1).to(torch.float64)      # promote to float64
    B, Sx, Sy, Sz = u.shape
    device = u.device

    # physical k like MATLAB
    def _k_axes_phys64(S, dx):
        L = S * dx
        half = S // 2
        kv = torch.cat([torch.arange(0, half + 1, device=device),
                        torch.arange(-half + 1, 0, device=device)], dim=0).to(torch.float64)
        return (2.0 * np.pi / L) * kv

    kx = _k_axes_phys64(Sx, dx); ky = _k_axes_phys64(Sy, dx); kz = _k_axes_phys64(Sz, dx)
    PX, QY, RZ = torch.meshgrid(1j*kx, 1j*ky, 1j*kz, indexing='ij')
    K2 = (PX/(1j))**2 + (QY/(1j))**2 + (RZ/(1j))**2  # real, >=0 (float64)

    U = torch.fft.fftn(u, dim=(1,2,3))

    fx = torch.fft.ifftn(PX * U, dim=(1,2,3)).real
    fy = torch.fft.ifftn(QY * U, dim=(1,2,3)).real
    fz = torch.fft.ifftn(RZ * U, dim=(1,2,3)).real

    s2 = fx*fx + fy*fy + fz*fz
    s2 = _cap_s2(s2)  # <--- important OOD stabilizer

    f1 = s2 * fx; f2 = s2 * fy; f3 = s2 * fz
    div_hat = PX * torch.fft.fftn(f1, dim=(1,2,3)) \
            + QY * torch.fft.fftn(f2, dim=(1,2,3)) \
            + RZ * torch.fft.fftn(f3, dim=(1,2,3))

    s_hat = torch.fft.fftn(u / dt, dim=(1,2,3)) + div_hat
    denom = (1.0 / dt) - K2 + float(eps) * (K2 ** 2)

    v_hat = s_hat / (denom + 1e-16)  # slightly larger epsilon in 64-bit
    up = torch.fft.ifftn(v_hat, dim=(1,2,3)).real
    return up.to(torch.float32).unsqueeze(-1)   # return in float32

def _scheme_residual_fourier_mbe(u0, up, dt, dx, eps):
    """
    Semi-implicit residual in Fourier matching semi_implicit_step_mbe:
      R̂ = [1/dt - k^2 + eps k^4] * Û^{n+1} - { FFT(u^n/dt) + i k · FFT((|∇u^n|^2 ∇u^n)) }
    Precondition by denom to whiten spectrum, then L2 mean.
    """
    u0 = u0.float(); up = up.float()
    B, S, _, _ = u0.shape
    device, dtype = u0.device, u0.dtype

    kv = _k_axes_phys(S, dx, device, dtype)
    PX, QY, RZ = torch.meshgrid(1j*kv, 1j*kv, 1j*kv, indexing='ij')
    K2 = (PX/(1j))**2 + (QY/(1j))**2 + (RZ/(1j))**2

    U0 = torch.fft.fftn(u0, dim=(1,2,3))
    UP = torch.fft.fftn(up, dim=(1,2,3))

    fx = torch.fft.ifftn(PX * U0, dim=(1,2,3)).real
    fy = torch.fft.ifftn(QY * U0, dim=(1,2,3)).real
    fz = torch.fft.ifftn(RZ * U0, dim=(1,2,3)).real
    s2 = fx*fx + fy*fy + fz*fz
    s2 = _cap_s2(s2)
    f1, f2, f3 = s2*fx, s2*fy, s2*fz
    div_hat = PX * torch.fft.fftn(f1, dim=(1,2,3)) \
            + QY * torch.fft.fftn(f2, dim=(1,2,3)) \
            + RZ * torch.fft.fftn(f3, dim=(1,2,3))

    denom = (1.0/dt) - K2 + float(eps) * (K2**2)
    rhs   = (1.0/dt) * U0 + div_hat
    rhat  = denom * UP - rhs
    rhat  = rhat / (denom + 1e-12)

    r2 = rhat.real**2 + rhat.imag**2
    return r2.mean()

def physics_collocation_tau_L2_MBE(u_in, u_pred,
                                   tau=0.5 - 1.0/(2.0*_math.sqrt(5.0)),
                                   normalize=True):
    """
    MBE collocation (L2) at u_tau:
      R_tau = (u^{n+1}-u^n)/dt - [ -Δu_tau - eps ∇^4 u_tau + ∇·(|∇u_tau|^2 ∇u_tau) ].
    """
    assert config.PROBLEM == 'MBE3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    ut    = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    ux, uy, uz = grad_fourier(u_tau, dx)
    s2 = _cap_s2(ux * ux + uy * uy + uz * uz)  # <--- cap here too
    #s2 = ux*ux + uy*uy + uz*uz
    # (optional) gentle clamp to avoid rare spiky batches; remove if undesired:
    # s2 = torch.clamp(s2, max=36.0)

    vx, vy, vz = s2*ux, s2*uy, s2*uz
    div_term   = div_fourier(vx, vy, vz, dx)
    lap_u      = laplacian_fourier_3d_phys(u_tau, dx)
    bi_u       = biharmonic(u_tau, dx)

    rhs_tau = -lap_u - eps * bi_u + div_term

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau
    return (R**2).mean().to(u_pred.dtype)




def mbe_free_energy_density(u, dx, eps):
    """
    Slope-selection MBE energy density:
        f = (eps/2) |Δu|^2 + 1/4 (|∇u|^2 - 1)^2    (constant shift ignored).
    """
    grad2 = _periodic_grad2(u, dx)       # |∇u|^2
    lap   = laplacian_fourier_3d_phys(u, dx)
    return 0.5 * eps * (lap * lap) + 0.25 * (grad2 - 1.0) * (grad2 - 1.0)


def energy_penalty_mbe(u_in, u_pred, dx, eps):
    """One-sided hinge: penalize F(u^{n+1}) > F(u^n)."""
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    F0 = mbe_free_energy_density(u0, dx, eps).mean(dim=(1,2,3))
    Fp = mbe_free_energy_density(up, dx, eps).mean(dim=(1,2,3))
    return torch.relu(Fp - F0).mean()
# ==================

def semi_implicit_step_ch(u_in, dt, dx, eps):
    """
    CORRECTED to exactly match MATLAB:
        (1 + dt*(2k² + ε²k⁴)) û^{n+1} = û^n - dt*k² * FFT(u^n³ - 3u^n)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)²|k|²

    # Exact MATLAB match
    chem0_hat = torch.fft.fftn(u0 ** 3 - 3.0 * u0, dim=(1, 2, 3))
    U0 = torch.fft.fftn(u0, dim=(1, 2, 3))

    denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))
    numer = U0 - dt * k2 * chem0_hat  # Note: k2 here is (kxx+kyy+kzz) equivalent

    U1 = numer / (denom + 1e-12)
    u1 = torch.fft.ifftn(U1, dim=(1, 2, 3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)

def semi_implicit_step_ch_T_out(u_in, dt, dx, eps):
    """
    CORRECTED to exactly match MATLAB:
        (1 + dt*(2k² + ε²k⁴)) û^{n+1} = û^n - dt*k² * FFT(u^n³ - 3u^n)

    Multi-step extension:
      - If T_out == 1: original one-step behavior.
      - If T_out > 1: returns (B,S,S,S,T_out) with T_out semi-implicit steps.
    """
    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)²|k|²

    # read T_out from config (support either T_OUT or T_out name)
    T_out = int(getattr(config, "T_OUT", getattr(config, "T_out", 1)))

    # ---------- single-step: original behavior ----------
    if T_out <= 1:
        chem0_hat = torch.fft.fftn(u0 ** 3 - 3.0 * u0, dim=(1, 2, 3))
        U0 = torch.fft.fftn(u0, dim=(1, 2, 3))

        denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))
        numer = U0 - dt * k2 * chem0_hat

        U1 = numer / (denom + 1e-12)
        u1 = torch.fft.ifftn(U1, dim=(1, 2, 3)).real
        return u1.unsqueeze(-1).to(u_in.dtype)  # (B,S,S,S,1)

    # ---------- multi-step: iterate T_out times ----------
    steps = torch.empty(B, S, S, S, T_out, device=u0.device, dtype=u0.dtype)
    u_curr = u0

    # denom does not depend on u, only on k2
    denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))

    for t in range(T_out):
        chem_hat = torch.fft.fftn(u_curr ** 3 - 3.0 * u_curr, dim=(1, 2, 3))
        U0 = torch.fft.fftn(u_curr, dim=(1, 2, 3))

        numer = U0 - dt * k2 * chem_hat
        U1 = numer / (denom + 1e-12)
        u_next = torch.fft.ifftn(U1, dim=(1, 2, 3)).real

        steps[..., t] = u_next
        u_curr = u_next

    # (B,S,S,S,T_out), no extra singleton channel, matches TNO output
    return steps.to(u_in.dtype)

def _rhs_ch3d(u, dx, eps):
    """
    CORRECTED CH3D RHS to match MATLAB generator:
        u_t = -Δ[2u + ε²Δu + (u³ - 3u)]
    This matches the MATLAB semi-implicit splitting.
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)  # Δu
    bi_u = biharmonic(u, dx)  # Δ²u
    chem = u ** 3 - 3.0 * u  # f'(u) = u³ - 3u

    # The RHS is -Δ of everything
    return -laplacian_fourier_3d_phys(2.0 * u + (eps ** 2) * lap_u + chem, dx)

import math as _m

def physics_collocation_tau_L2_CH(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*_m.sqrt(5.0)),
                                  normalize=True):
    """
    CH3D L2 collocation at u_tau (match SH/PFC/MBE structure):
      R_tau = (u^{n+1}-u^n)/dt - RHS_CH(u_tau)
    scored in **L2** with optional per-sample normalization.
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float()
    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    rhs_tau = _rhs_ch3d(u_tau, dx, eps)

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R**2).mean().to(u_pred.dtype)


def physics_collocation_tau_L2_CH_T_out(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*_m.sqrt(5.0)),
                                  normalize=True):
    """
    CH3D L2 collocation at u_tau (now supports multi-step T_out):
      R_tau = (u^{n+1}-u^n)/dt - RHS_CH(u_tau)
    If u_pred has shape (B,S,S,S,T_out), we apply the same formula
    for each predicted step, broadcasting u^n.
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float() # (B,S,S,S) or (B,S,S,S,T)

    # ----- single-step (backward compatible) -----
    if up.dim() == 4:
        ut   = (up - u0) / dt              # (B,S,S,S)
        u_tau = (1.0 - tau) * u0 + tau * up
        rhs_tau = _rhs_ch3d(u_tau, dx, eps)  # (B,S,S,S)

        if normalize:
            s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau
        return (R**2).mean().to(u_pred.dtype)

    # ----- multi-step: up: (B,S,S,S,T_out) -----
    assert up.dim() == 5, "u_pred must be 4D or 5D"

    B, Sx, Sy, Sz, T = up.shape
    u0_exp = u0.unsqueeze(-1)                  # (B,S,S,S,1)

    ut    = (up - u0_exp) / dt                 # (B,S,S,S,T)
    u_tau = (1.0 - tau) * u0_exp + tau * up    # (B,S,S,S,T)

    # RHS_CH expects (B,S,S,S), so flatten time into batch:
    u_tau_flat = u_tau.permute(0,4,1,2,3).reshape(B*T, Sx, Sy, Sz)
    rhs_flat   = _rhs_ch3d(u_tau_flat, dx, eps)           # (B*T,S,S,S)
    rhs_tau    = rhs_flat.view(B, T, Sx, Sy, Sz).permute(0,2,3,4,1)  # (B,S,S,S,T)

    if normalize:
        # normalize over all spatial + time dims
        s_t = ut.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R**2).mean().to(u_pred.dtype)
