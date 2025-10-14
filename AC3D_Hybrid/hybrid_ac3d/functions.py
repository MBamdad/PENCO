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

def physics_residual_midpoint_ch_Hm1(u_in, u_pred):
    """
    CH3D ONLY:
      R_m = (u^{n+1}-u^n)/dt - Δ μ(u_m), with μ(u)=u^3 - 3u - ε^2 Δ u
      Loss = ||R_m||_{H^{-1}}^2  ≈  Σ |R_hat|^2 / k^2
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float()
    um = 0.5 * (u0 + up)

    # ut
    ut = (up - u0) / dt

    # μ(u_m) = (u^3 - 3u) - ε^2 Δ u   (dealised cubic for stability)
    um_da = dealias_two_thirds(um)
    chem_m = (um_da**3 - 3.0*um_da)
    mu_m = chem_m - (eps**2) * laplacian_fourier_3d_phys(um, dx)

    # Δ μ(u_m)
    lap_mu_m = laplacian_fourier_3d_phys(mu_m, dx)

    # residual
    Rm = ut - lap_mu_m                        # (B,S,S,S)

    # H^{-1} norm: sum |R_hat|^2 / k^2  (avoid k=0 blow-up with tiny floor)
    B, S, _, _ = Rm.shape
    k2, inv_k2 = _fft_rk2(S, S, S, dx, Rm.device, dtype=torch.float32)  # (S,S,S)
    Rm_hat = torch.fft.fftn(Rm, dim=(1,2,3))                            # complex
    # power spectrum
    pow_spec = (Rm_hat.real**2 + Rm_hat.imag**2)

    weighted = pow_spec * inv_k2   # divide by k^2  (H^{-1})

    # Mean over all modes and batch; robustify a bit (Charbonnier)
    loss = _charbonnier_mean(weighted)
    return loss.to(u_pred.dtype)

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


# --- exact MATLAB-matched CH semi-implicit step (double precision internal) ---
def _semi_implicit_step_ch(u_in, dt, dx, eps, stab_coef=2.0, use_dealias=False):
    """
    CH3D teacher step, arithmetic matched to the MATLAB generator:
      s_hat = FFT(u^n) - dt * k^2 * FFT( (u^n)^3 - 3 u^n )
      u_hat^{n+1} = s_hat / (1 + dt * (2 k^2 + eps^2 k^4))
    Notes:
      - internal math in float64 for reproducibility
      - k-grid uses the same [0:N/2, -N/2+1:-1] convention as MATLAB
      - no dealiasing (dataset has none)
    """
    assert config.PROBLEM == 'CH3D'
    u0 = u_in.squeeze(-1)
    B, S, _, _ = u0.shape

    # work in float64 to reduce roundoff; cast back on return
    u0w = u0.to(torch.float64)

    # MATLAB-consistent physical k-grid
    k2, k4 = _kgrid(S, dx, u0w.device)
    k2 = k2.to(torch.float64); k4 = k4.to(torch.float64)

    # chemistry at u^n (dataset)
    chem0w = (u0w * u0w * u0w) - 3.0 * u0w

    u0_hat   = torch.fft.fftn(u0w,   dim=(1,2,3))
    chem0_hat= torch.fft.fftn(chem0w,dim=(1,2,3))

    s_hat  = u0_hat - dt * k2 * chem0_hat
    denom  = 1.0 + dt * (2.0 * k2 + (eps * eps) * k4)

    u1_hat = s_hat / denom
    u1     = torch.fft.ifftn(u1_hat, dim=(1,2,3)).real

    return u1.to(u_in.dtype).unsqueeze(-1)

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
    u3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))  # no dealias (dataset)

    denom  = (1.0/dt) + (1.0 - eps) + k4
    numer  = (1.0/dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat

    u1_hat = numer / denom
    u1     = torch.fft.ifftn(u1_hat, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)

'''''
def physics_guided_update_sh_optimal(u_in, y_pred, alpha_cap=1.0):
    """
    SH3D ONLY.
    Find alpha* that minimizes the semi-implicit teacher residual (in Fourier)
    along the line: u_alpha = up + alpha (u_si - up), alpha in [0, alpha_cap].
    Returns blended field with stop-on-mass (no mass constraint for SH).
    """
    assert config.PROBLEM == 'SH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0  = u_in.squeeze(-1).float()
    up  = y_pred.squeeze(-1).float()
    usi = semi_implicit_step_sh(u_in, dt, dx, eps).squeeze(-1).float()

    B, S, _, _ = up.shape
    k2 = _k_spectrum(S, S, S, dx, up.device)
    k4 = k2 * k2

    u0_hat  = torch.fft.fftn(u0, dim=(1,2,3))
    up_hat  = torch.fft.fftn(up, dim=(1,2,3))
    usi_hat = torch.fft.fftn(usi, dim=(1,2,3))
    u3_hat  = torch.fft.fftn(u0**3, dim=(1,2,3))  # dataset-consistent: no dealias

    denom   = (1.0/dt) + (1.0 - eps) + k4
    rhs_hat = (1.0/dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat

    r0_hat = denom * up_hat  - rhs_hat           # residual at alpha=0
    a_hat  = denom * (usi_hat - up_hat)          # direction

    # weighted LS in Fourier (Parseval), same spirit as scheme_residual
    w = 1.0 / (denom + 1e-12)                   # preconditioning to avoid high-k domination
    def wdot(A, B):
        return (w * (A.real*B.real + A.imag*B.imag)).sum(dim=(1,2,3))

    with torch.no_grad():
        c1 = wdot(r0_hat, a_hat)                                        # (B,)
        c2 = (w * (a_hat.real**2 + a_hat.imag**2)).sum((1,2,3)) + 1e-24 # (B,)
        alpha_star = (-c1 / c2).clamp(0.0, alpha_cap).view(B,1,1,1)

    u_blend = up + alpha_star * (usi - up)
    return u_blend.unsqueeze(-1).to(y_pred.dtype)
'''

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


# --- CH projection with mixed chemistry (uses the same MATLAB-matched kernel) ---
def si_projection_ch_mixed(u_in, u_pred, dt, dx, eps, tau=0.5):
    """
    Semi-implicit projection using mixed chemistry:
      u_m = (1-τ) u^n + τ u^{n+1}_pred
      û^{n+1}_proj = [ û^n - dt * k^2 * FFT( u_m^3 - 3 u_m ) ] / [ 1 + dt (2 k^2 + eps^2 k^4) ]
    Computed in float64 with MATLAB k-grid; no dealiasing (to match dataset).
    """
    assert config.PROBLEM == 'CH3D'
    u0 = u_in.squeeze(-1).to(torch.float64)
    up = u_pred.squeeze(-1).to(torch.float64)
    um = (1.0 - tau) * u0 + tau * up

    B, S, _, _ = u0.shape
    k2, k4 = _kgrid(S, dx, u0.device)
    k2 = k2.to(torch.float64); k4 = k4.to(torch.float64)

    u0_hat   = torch.fft.fftn(u0, dim=(1,2,3))
    chem_hat = torch.fft.fftn(um*um*um - 3.0*um, dim=(1,2,3))

    denom  = 1.0 + dt * (2.0 * k2 + (eps * eps) * k4)
    up_hat = (u0_hat - dt * k2 * chem_hat) / denom

    u_proj = torch.fft.ifftn(up_hat, dim=(1,2,3)).real
    return u_proj.unsqueeze(-1).to(u_pred.dtype)


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

# --- MATLAB-consistent μ (NO dealias), to match your dataset exactly ---
def mu_ch_matlab(u, dx, eps):
    return (u**3 - 3.0*u) - (eps**2) * laplacian_fourier_3d_phys(u, dx)

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


def physics_residual_midpoint_normalized_ch(u_in, u_pred):
    """
    CH midpoint residual with per-batch normalization and H^{-1} metric,
    focused on interfaces via a soft weight.
      R_m = (u^{n+1}-u^n)/dt - Δ μ(u_m),   μ(u)=u^3 - 3u - ε^2 Δu
    We compute:  R̃ = (u_t / ||u_t||) - (Δμ / ||Δμ||), then H^{-1} norm of w * R̃.
    """
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    um = 0.5 * (u0 + up)

    ut = (up - u0) / dt
    mu_m = (um**3 - 3.0*um) - (eps**2) * laplacian_fourier_3d_phys(um, dx)
    lap_mu_m = laplacian_fourier_3d_phys(mu_m, dx)

    # per-sample normalization so ut and Δμ carry equal weight
    s_t  = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    s_mu = lap_mu_m.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    Rm = ut / s_t - lap_mu_m / s_mu

    # emphasize interfaces
    w = interface_weight(um, dx)  # [0,1]
    Rm = Rm * w

    # H^{-1} metric: ||R||_{H^{-1}}^2 ≈ ∑ |R̂|^2 / k^2   (with tiny floor)
    B, S, _, _ = Rm.shape
    k2, inv_k2 = _fft_rk2(S, S, S, dx, Rm.device, dtype=torch.float32)
    Rm_hat = torch.fft.fftn(Rm, dim=(1,2,3))
    r2 = (Rm_hat.real**2 + Rm_hat.imag**2) * inv_k2
    return torch.sqrt(r2 + 1e-8).mean().to(u_pred.dtype)


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
# -------------------------
def _rhs_ac3d(u, dx, eps2):
    # Δu - (u^3 - u)/eps^2
    lap_u = laplacian_fourier_3d_phys(u, dx)
    u_nl = dealias_two_thirds(u)
    return lap_u - (1.0/eps2) * (u_nl**3 - u_nl)

def _rhs_ch3d(u, dx, eps):
    # OLD:
    # u_da = dealias_two_thirds(u)
    # chem = (u_da**3 - 3.0*u_da)
    # return laplacian_fourier_3d_phys(chem, dx) - (eps**2) * biharmonic(u, dx)

    # NEW (system form; still mathematically identical):
    mu = mu_ch(u, dx, eps)
    return laplacian_fourier_3d_phys(mu, dx)


def _rhs_mbe3d(u, dx, eps):
    # ∂t u = -∇·(|∇u|^2 ∇u) + ε ∇⁴ u
    ux, uy, uz = grad_fourier(u, dx)
    s = ux*ux + uy*uy + uz*uz
    vx, vy, vz = s*ux, s*uy, s*uz
    div_term = div_fourier(vx, vy, vz, dx)
    return -div_term + eps * biharmonic(u, dx)

def _rhs_pfc3d(u, dx, eps):
    # ∂t u = (1 - ε)∇²u + ∇⁴u + ∇⁶u - ∇²(u^3)
    lap_u = laplacian_fourier_3d_phys(u, dx)
    bi_u  = biharmonic(u, dx)
    tri_u = triharmonic(u, dx)
    u_da = dealias_two_thirds(u)
    lap_u3 = laplacian_fourier_3d_phys(u_da**3, dx)
    return (1.0 - eps) * lap_u + bi_u + tri_u - lap_u3

def _rhs_sh3d(u, dx, eps):
    """
    Swift–Hohenberg (dataset-consistent split):
        u_t = (1-ε) u - 2 Δ u - Δ^2 u - u^3
    NOTE:
      • NO dealiasing on u^3 (matches the MATLAB generator).
      • Clamp inside the cubic to avoid overflow when the net overshoots early.
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)     # Δu
    bi_u  = biharmonic(u, dx)                    # Δ^2 u
    u_safe = torch.clamp(u, -5.0, 5.0)           # safe cubic; no in-place to keep graph
    return (1.0 - eps) * u - 2.0 * lap_u - bi_u - (u_safe ** 3)



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
    elif config.PROBLEM == 'CH3D':
        return _semi_implicit_step_ch(u_in, dt, dx, eps=np.sqrt(eps2), stab_coef=2.0, use_dealias=False)
    elif P == 'SH3D':  # <-- NEW
        return _semi_implicit_step_sh(u_in, dt, dx, eps=config.EPSILON_PARAM)

    else:
        # fallback (unchanged) explicit Euler
        u0 = u_in.squeeze(-1)
        rhs = pde_rhs(u0, dx, config.EPSILON_PARAM)
        return (u0 + dt * rhs).unsqueeze(-1)

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

# --- add near your other helpers ---
def mu_ch(u, dx, eps):
    """
    Chemical potential for CH: μ = u^3 - 3u - ε^2 ∇²u
    Dealias cubic to avoid spectral folding.
    """
    u_da = dealias_two_thirds(u)
    return (u_da**3 - 3.0*u_da) - (eps**2) * laplacian_fourier_3d_phys(u, dx)


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
        u3_hat = torch.fft.fftn(u0 ** 3, dim=(1, 2, 3))  # dataset-consistent (no dealias)

        eps = config.EPSILON_PARAM
        denom = (1.0 / dt) + (1.0 - eps) + k4

        # residual in Fourier of the semi-implicit update
        rhs_hat = (1.0 / dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat
        rhat = denom * up_hat - rhs_hat

        # precondition to avoid high-k domination (same idea as CH)
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
def physics_residual_midpoint_normalized_ch_direct(u_in, u_pred):
    """
    CH midpoint residual with per-batch normalization in an H^{-1} metric,
    using the DIRECT PDE (no μ):
        R_m = (u^{n+1}-u^n)/dt - [Δ(u_m^3-3u_m) - eps^2 ∇^4 u_m],  u_m=(u^n+u^{n+1})/2
    """
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    um = 0.5 * (u0 + up)

    ut    = (up - u0) / dt
    rhs_m = laplacian_fourier_3d_phys(um**3 - 3.0*um, dx) - (eps**2) * biharmonic(um, dx)

    # per-sample normalization (balances ut vs RHS)
    s_t  = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    s_r  = rhs_m.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    Rm = ut / s_t - rhs_m / s_r

    # H^{-1} metric: ||R||_{H^{-1}}^2 ≈ Σ |R̂|^2 / (k^2 + ε0)
    B, S, _, _ = Rm.shape
    k2, inv_k2 = _fft_rk2(S, S, S, dx, Rm.device, dtype=torch.float32)
    Rm_hat = torch.fft.fftn(Rm, dim=(1,2,3))
    r2 = (Rm_hat.real**2 + Rm_hat.imag**2) * inv_k2
    return torch.sqrt(r2 + 1e-8).mean().to(u_pred.dtype)

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

def physics_collocation_tau_Hm1_CH_direct(u_in, u_pred, tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                          normalize=True, use_interface_weight=True):
    """
    CH3D: R_tau = (u^{n+1}-u^n)/dt - [Δ(u_tau^3-3u_tau) - eps^2 ∇^4 u_tau],
    with u_tau = (1-τ)u^n + τ u^{n+1}. Score in H^{-1}.
    - No de-aliasing (matches dataset).
    - Per-batch normalization balances ut vs RHS.
    - Optional soft interface weighting.
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    ut = (up - u0) / dt

    um = (1.0 - tau) * u0 + tau * up
    rhs_tau = laplacian_fourier_3d_phys(um**3 - 3.0*um, dx) - (eps**2) * biharmonic(um, dx)

    if normalize:
        s_t = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    if use_interface_weight:
        w = interface_weight(um, dx)  # already defined in your file
        R = R * w

    return _hm1_mean(R, dx)  # uses your existing _fft_rk2-based helper


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



def si_projection_ch_mixed(u_in, u_pred, dt, dx, eps, tau=0.5):
    """
    Semi-implicit CH projection using mixed chemistry:
      u_m = (1-τ) u^n + τ u^{n+1}_pred
      û^{n+1}_proj = [ û^n - dt * k^2 * FFT( u_m^3 - 3 u_m ) ] / [ 1 + dt (2 k^2 + eps^2 k^4) ]
    No dealiasing (matches dataset). Shapes: (B,S,S,S,1) in/out.
    """
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    um = (1.0 - tau) * u0 + tau * up

    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)^2 |k|^2
    k4 = k2 * k2

    u0_hat   = torch.fft.fftn(u0, dim=(1,2,3))
    chem_hat = torch.fft.fftn(um**3 - 3.0*um, dim=(1,2,3))

    denom = 1.0 + dt * (2.0 * k2 + (eps*eps) * k4)
    up_hat = (u0_hat - dt * k2 * chem_hat) / denom

    u_proj = torch.fft.ifftn(up_hat, dim=(1,2,3)).real
    return u_proj.unsqueeze(-1)

def physics_guided_update_ch(u_in, y_pred, alpha=0.35, tau=0.5):
    """
    Blend the semi-implicit projection back into the prediction:
      y_phys = y_pred + alpha * (u_proj - y_pred)
    then mass-project to enforce CH conservation.
    """
    u_proj = si_projection_ch_mixed(u_in, y_pred, config.DT, config.DX, config.EPSILON_PARAM, tau=tau)
    y_phys = y_pred + alpha * (u_proj - y_pred)
    y_phys = mass_project_pred(y_phys, u_in)  # keep total mass exactly
    return y_phys

# === Optimal Physics-Guided Update for CH3D (minimizes SI residual in H^{-1}) ===
import torch, math
import torch.nn.functional as F
import numpy as np
import config

def si_projection_ch_mixed(u_in, u_pred, dt, dx, eps, tau=0.5):
    """
    Semi-implicit projection using mixed chemistry (dataset-consistent, no dealias).
    Inputs/outputs: (B,S,S,S,1).
    """
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    um = (1.0 - tau) * u0 + tau * up

    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)^2 |k|^2
    k4 = k2 * k2

    u0_hat   = torch.fft.fftn(u0, dim=(1,2,3))
    chem_hat = torch.fft.fftn(um**3 - 3.0*um, dim=(1,2,3))

    denom = 1.0 + dt * (2.0 * k2 + (eps*eps) * k4)
    up_hat = (u0_hat - dt * k2 * chem_hat) / denom

    u_proj = torch.fft.ifftn(up_hat, dim=(1,2,3)).real
    return u_proj.unsqueeze(-1).to(u_pred.dtype)

def physics_guided_update_ch_optimal(u_in, y_pred, tau=0.0, alpha_cap=1.0,
                                     low_k_snap_frac=0.45, use_float64=True):
    """
    CH3D ONLY.
    Physics-guided update with *low-k spectral snap* + optimal H^{-1} blend.

    Steps:
      1) Semi-implicit projection using mixed chemistry (tau) -> u_proj
      2) In Fourier space, replace all modes with radius <= (low_k_snap_frac * r_max)
         by those of u_proj (dataset teacher). High-k stay from the net.
      3) Choose alpha* in [0, alpha_cap] that minimizes the *teacher* residual
         (semi-implicit) under H^{-1} weighting (as before).
      4) Mass-project the result to enforce conservation.

    Notes:
      - Defaults: tau=0.0 (exact dataset chemistry), alpha_cap=1.0 (allow full correction).
      - Works entirely inside this function; call sites do not need to change.
    """
    assert config.PROBLEM == 'CH3D', "physics_guided_update_ch_optimal is CH3D-specific."

    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    # ---- inputs ----
    u0 = u_in.squeeze(-1)
    up = y_pred.squeeze(-1)

    # higher precision for spectral math (optional)
    work_dtype = torch.float64 if (use_float64 and up.dtype == torch.float32) else up.dtype
    u0w = u0.to(work_dtype)
    upw = up.to(work_dtype)

    # 1) dataset-consistent semi-implicit projection (mixed chemistry)
    u_proj = si_projection_ch_mixed(u_in, y_pred, dt, dx, eps, tau=tau).squeeze(-1).to(work_dtype)

    # 2) low-k spectral snap: replace low-frequency modes by teacher's modes
    B, S, _, _ = upw.shape
    up_hat    = torch.fft.fftn(upw,    dim=(1,2,3))
    uproj_hat = torch.fft.fftn(u_proj, dim=(1,2,3))

    fx = torch.fft.fftfreq(S, d=1.0).to(upw.device)
    fy = torch.fft.fftfreq(S, d=1.0).to(upw.device)
    fz = torch.fft.fftfreq(S, d=1.0).to(upw.device)
    FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    rmax = r.max()
    low_mask = (r <= low_k_snap_frac * rmax).to(up_hat.dtype)

    # snap low-k to teacher, keep high-k from the net
    usnap_hat = low_mask * uproj_hat + (1.0 - low_mask) * up_hat
    u_snap = torch.fft.ifftn(usnap_hat, dim=(1,2,3)).real  # (B,S,S,S)

    # 3) choose optimal alpha* (H^{-1} teacher residual) between up and u_snap
    k2 = _k_spectrum(S, S, S, dx, upw.device).to(work_dtype)
    k4 = k2 * k2
    denom = 1.0 + dt * (2.0 * k2 + (eps * eps) * k4)

    u0_hat    = torch.fft.fftn(u0w, dim=(1,2,3))
    chem0_hat = torch.fft.fftn(u0w**3 - 3.0*u0w, dim=(1,2,3))
    const_hat = u0_hat - dt * k2 * chem0_hat

    up_hat     = torch.fft.fftn(upw,    dim=(1,2,3))
    usnap_hat2 = torch.fft.fftn(u_snap, dim=(1,2,3))

    r0_hat = denom * up_hat - const_hat                  # residual at alpha=0
    a_hat  = denom * (usnap_hat2 - up_hat)               # direction when increasing alpha

    # H^{-1} weights
    _, inv_k2 = _fft_rk2(S, S, S, dx, upw.device, dtype=work_dtype)
    w = inv_k2

    def wdot(A, B):
        return (w * (A.real * B.real + A.imag * B.imag)).sum(dim=(1,2,3))

    with torch.no_grad():
        c1 = wdot(r0_hat, a_hat)                                          # (B,)
        c2 = (w * (a_hat.real**2 + a_hat.imag**2)).sum((1,2,3)) + 1e-24   # (B,)
        alpha_star = (-c1 / c2).clamp(0.0, alpha_cap).view(B, 1, 1, 1)

    y_phys = upw + alpha_star * (u_snap - upw)        # blend back from snapped low-k
    y_phys = y_phys.unsqueeze(-1)
    # 4) enforce mass exactly
    y_phys = mass_project_pred(y_phys.to(y_pred.dtype), u_in)
    return y_phys

import torch, math
import torch.nn.functional as F

def physics_guided_update_sh_optimal(u_in, y_pred, alpha_cap=0.7, low_k_snap_frac=0.48):
    """
    SH3D ONLY. Blend y_pred toward the semi-implicit teacher usi with optimal alpha*
    that MINIMIZES the semi-implicit residual (preconditioned). If the best alpha*
    would increase residual, do nothing (alpha=0).
    """
    assert config.PROBLEM == 'SH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0  = u_in.squeeze(-1).float()
    up  = y_pred.squeeze(-1).float()
    usi = semi_implicit_step_sh(u_in, dt, dx, eps).squeeze(-1).float()

    B, S, _, _ = up.shape
    device = up.device
    k2 = _k_spectrum(S, S, S, dx, device); k4 = k2 * k2
    denom = (1.0 / dt) + (1.0 - eps) + k4

    u0_hat  = torch.fft.fftn(u0, dim=(1,2,3))
    up_hat  = torch.fft.fftn(up, dim=(1,2,3))
    usi_hat = torch.fft.fftn(usi, dim=(1,2,3))
    u3_hat  = torch.fft.fftn(u0**3, dim=(1,2,3))
    rhs_hat = (1.0 / dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat

    # light low-k snap (improves coherence of large scales)
    fx = torch.fft.fftfreq(S, d=1.0).to(device)
    FY, FX, FZ = torch.meshgrid(fx, fx, fx, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    mask_low = (r <= low_k_snap_frac * r.max()).to(up_hat.dtype)
    up_hat = mask_low * usi_hat + (1.0 - mask_low) * up_hat

    r0_hat = denom * up_hat - rhs_hat              # residual at alpha=0
    a_hat  = denom * (usi_hat - up_hat)            # descent direction toward usi
    w = 1.0 / (denom + 1e-12)                      # whitening

    def wdot(A, B):
        return (w * (A.real*B.real + A.imag*B.imag)).sum(dim=(1,2,3))

    with torch.no_grad():
        c1 = wdot(r0_hat, a_hat)                                     # (B,)
        c2 = (w * (a_hat.real**2 + a_hat.imag**2)).sum((1,2,3)) + 1e-24
        alpha_star = (-c1 / c2).clamp(0.0, alpha_cap).view(B,1,1,1)

        # MONOTONE SAFETY: ensure residual does not increase
        up_blend_hat = up_hat + alpha_star * (usi_hat - up_hat)
        r_star_hat   = denom * up_blend_hat - rhs_hat
        # If ⟨r*, r*⟩_w > ⟨r0, r0⟩_w, set alpha to 0
        r0_w = (w * (r0_hat.real**2 + r0_hat.imag**2)).sum((1,2,3))
        rs_w = (w * (r_star_hat.real**2 + r_star_hat.imag**2)).sum((1,2,3))
        bad = (rs_w > r0_w).view(B,1,1,1)
        alpha_star = torch.where(bad, torch.zeros_like(alpha_star), alpha_star)

    y_blend = up + alpha_star * (usi - up)
    return y_blend.unsqueeze(-1).to(y_pred.dtype)
