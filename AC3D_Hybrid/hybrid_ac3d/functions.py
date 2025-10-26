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
# -------------------------
def _rhs_ac3d(u, dx, eps2):
    # Δu - (u^3 - u)/eps^2
    lap_u = laplacian_fourier_3d_phys(u, dx)
    u_nl = dealias_two_thirds(u)
    return lap_u - (1.0/eps2) * (u_nl**3 - u_nl)


def _rhs_mbe3d(u, dx, eps):
    """
    MBE3D (slope-selection, dataset-consistent):
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
    # match dataset: NO dealiasing in the cubic
    lap_u3 = laplacian_fourier_3d_phys(u**3, dx)
    return (1.0 - eps) * lap_u + 2.0 * bi_u + tri_u - lap_u3

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
    U3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))  # no dealias (dataset)

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
    U3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))    # dataset-consistent (no dealias)

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


#################


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
    # RHS that matches dataset (no dealias):
    rhs_tau = _rhs_pfc3d(u_tau, dx, eps)

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau
    return (R**2).mean().to(u_pred.dtype)


def physics_guided_update_pfc_optimal(u_in, y_pred, alpha_cap=0.6, low_k_snap_frac=0.45):
    """
    PFC3D ONLY. Blend y_pred toward the semi-implicit teacher (PFC) with
    optimal alpha* (Fourier-preconditioned LS on the semi-implicit residual),
    including a light low-k spectral snap to the teacher.
    """
    assert config.PROBLEM == 'PFC3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0  = u_in.squeeze(-1).float()
    up  = y_pred.squeeze(-1).float()
    usi = semi_implicit_step_pfc(u_in, dt, dx, eps).squeeze(-1).float()

    B, S, _, _ = up.shape
    k2 = _k_spectrum(S, S, S, dx, up.device)
    denom = (1.0/dt) + (1.0 - eps)*k2 + (k2**3)

    U0   = torch.fft.fftn(u0, dim=(1,2,3))
    UP   = torch.fft.fftn(up, dim=(1,2,3))
    USI  = torch.fft.fftn(usi, dim=(1,2,3))
    U3   = torch.fft.fftn(u0**3, dim=(1,2,3))  # dataset-consistent

    rhs  = (1.0/dt)*U0 - k2*U3 + 2.0*(k2**2)*U0

    # light low-k snap
    fx = torch.fft.fftfreq(S, d=1.0).to(up.device)
    FX, FY, FZ = torch.meshgrid(fx, fx, fx, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    mask_low = (r <= low_k_snap_frac * r.max()).to(UP.dtype)
    UP = mask_low * USI + (1.0 - mask_low) * UP

    r0 = denom * UP  - rhs
    a  = denom * (USI - UP)
    w  = 1.0 / (denom + 1e-12)

    def wdot(A, B): return (w * (A.real*B.real + A.imag*B.imag)).sum(dim=(1,2,3))

    with torch.no_grad():
        c1 = wdot(r0, a)
        c2 = (w * (a.real**2 + a.imag**2)).sum((1,2,3)) + 1e-24
        alpha = (-c1 / c2).clamp(0.0, alpha_cap).view(B,1,1,1)

        # monotone safety
        UP_star = UP + alpha * (USI - UP)
        rs = denom * UP_star - rhs
        r0_w = (w * (r0.real**2 + r0.imag**2)).sum((1,2,3))
        rs_w = (w * (rs.real**2 + rs.imag**2)).sum((1,2,3))
        alpha = torch.where((rs_w > r0_w).view(B,1,1,1),
                            torch.zeros_like(alpha), alpha)

    y_blend = up + alpha * (usi - up)
    return y_blend.unsqueeze(-1).to(y_pred.dtype)


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
'''''
def semi_implicit_step_mbe(u_in, dt, dx, eps):
    """
    One MBE3D semi-implicit step (matches the MATLAB generator):

      ŝ = FFT(u/dt) + [ p * FFT(f1) + q * FFT(f2) + r * FFT(f3) ]
      v̂ = ŝ / [ 1/dt - (k^2) + eps * (k^2)^2 ]
      u^{n+1} = IFFT(v̂)

    where p,q,r = i*2π/L * k components,
          fx = ∂_x u, etc,  f1 = (|∇u|^2) * fx, etc.
    Inputs:  u_in (B,S,S,S,1), dt, dx, eps
    Returns: (B,S,S,S,1)
    """
    u = u_in.squeeze(-1).float()              # (B,S,S,S)
    B, Sx, Sy, Sz = u.shape
    device, dtype = u.device, u.dtype

    # spectral axes (i*2π/L * k) as in MATLAB code
    kx = _k_axes_phys(Sx, dx, device, dtype)
    ky = _k_axes_phys(Sy, dx, device, dtype)
    kz = _k_axes_phys(Sz, dx, device, dtype)
    px, qy, rz = 1j * kx, 1j * ky, 1j * kz

    # grids
    PX, QY, RZ = torch.meshgrid(px, qy, rz, indexing='ij')
    K2 = (PX/(1j))**2 + (QY/(1j))**2 + (RZ/(1j))**2   # (2π/L)^2 |k|^2, real >=0

    U = torch.fft.fftn(u, dim=(1,2,3))

    # gradients in real space via spectral
    fx = torch.fft.ifftn(PX * U, dim=(1,2,3)).real
    fy = torch.fft.ifftn(QY * U, dim=(1,2,3)).real
    fz = torch.fft.ifftn(RZ * U, dim=(1,2,3)).real
    s2 = fx*fx + fy*fy + fz*fz

    f1 = s2 * fx
    f2 = s2 * fy
    f3 = s2 * fz

    # divergence in Fourier: i k·FFT(f)  (here already have p/q/r = i*2π/L * k)
    div_hat = PX * torch.fft.fftn(f1, dim=(1,2,3)) \
            + QY * torch.fft.fftn(f2, dim=(1,2,3)) \
            + RZ * torch.fft.fftn(f3, dim=(1,2,3))

    s_hat = torch.fft.fftn(u / dt, dim=(1,2,3)) + div_hat
    denom = (1.0 / dt) - K2 + (float(eps)) * (K2 ** 2)    # 1/dt - k^2 + eps k^4

    v_hat = s_hat / (denom + 1e-12)
    up = torch.fft.ifftn(v_hat, dim=(1,2,3)).real
    return up.unsqueeze(-1).to(u_in.dtype)
'''


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
# =============================================================================

def physics_guided_update_mbe_optimal(u_in, y_pred, alpha_cap=0.6, low_k_snap_frac=0.45):
    """
    MBE3D ONLY. Blend y_pred toward the semi-implicit MBE teacher with optimal alpha*.
    Matches your MATLAB step:
        (1/dt - k^2 + eps k^4) û^{n+1} = (1/dt) û^n + FFT(div(|∇u^n|^2 ∇u^n))
    We minimize the preconditioned semi-implicit residual in Fourier, with:
      • light low-k snap to the teacher
      • weighted LS alpha* in [0, alpha_cap]
      • monotone safety (don’t increase residual)
    """
    assert config.PROBLEM == 'MBE3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    # real-space fields
    u0  = u_in.squeeze(-1).float()   # (B,S,S,S)
    up  = y_pred.squeeze(-1).float()

    # teacher step (your MBE semi-implicit)
    usi = semi_implicit_step_mbe(u_in, dt, dx, eps).squeeze(-1).float()

    B, S, _, _ = up.shape
    device = up.device

    # k^2 spectrum and operator denominator
    k2 = _k_spectrum(S, S, S, dx, device)       # (2π/L)^2 |k|^2  >= 0
    denom = (1.0/dt) - k2 + eps * (k2**2)       # matches MATLAB denom: 1/dt - k^2 + eps k^4

    # ----- RHS(u^n) in real space: (1/dt) u^n + div(|∇u^n|^2 ∇u^n) -----
    ux, uy, uz = grad_fourier(u0, dx)
    s = ux*ux + uy*uy + uz*uz
    f1, f2, f3 = s*ux, s*uy, s*uz
    div_term = div_fourier(f1, f2, f3, dx)
    rhs = (1.0/dt) * u0 + div_term

    # FFTs
    U0   = torch.fft.fftn(u0,  dim=(1,2,3))
    UP   = torch.fft.fftn(up,  dim=(1,2,3))
    USI  = torch.fft.fftn(usi, dim=(1,2,3))
    RHS  = torch.fft.fftn(rhs, dim=(1,2,3))

    # ----- light low-k snap to teacher (improves large-scale coherence) -----
    fx = torch.fft.fftfreq(S, d=1.0).to(device)
    FX, FY, FZ = torch.meshgrid(fx, fx, fx, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    mask_low = (r <= low_k_snap_frac * r.max()).to(UP.dtype)
    UP = mask_low * USI + (1.0 - mask_low) * UP

    # ----- preconditioned semi-implicit residual (Parseval) -----
    r0 = denom * UP  - RHS                  # residual at alpha=0
    a  = denom * (USI - UP)                 # descent direction toward teacher
    w  = 1.0 / (denom + 1e-12)              # whitening to avoid high-k domination

    def wdot(A, B):
        return (w * (A.real*B.real + A.imag*B.imag)).sum(dim=(1,2,3))

    with torch.no_grad():
        c1 = wdot(r0, a)                                              # (B,)
        c2 = (w * (a.real**2 + a.imag**2)).sum((1,2,3)) + 1e-24       # (B,)
        alpha = (-c1 / c2).clamp(0.0, alpha_cap).view(B,1,1,1)

        # monotone safety: ensure residual doesn’t increase
        UP_star = UP + alpha * (USI - UP)
        rs = denom * UP_star - RHS
        r0_w = (w * (r0.real**2 + r0.imag**2)).sum((1,2,3))
        rs_w = (w * (rs.real**2 + rs.imag**2)).sum((1,2,3))
        alpha = torch.where((rs_w > r0_w).view(B,1,1,1),
                            torch.zeros_like(alpha), alpha)

    # blend back in real space
    y_blend = up + alpha * (usi - up)
    return y_blend.unsqueeze(-1).to(y_pred.dtype)


def physics_collocation_tau_Hm1_MBE(u_in, u_pred, tau=0.5 - 1.0/(2.0*_math.sqrt(5.0)),
                                    normalize=True):
    """
    MBE collocation scored in H^{-1} to emphasize conserved, low-k dynamics:
      R_tau = (u^{n+1}-u^n)/dt - [Δu_tau - eps ∇^4 u_tau + ∇·(|∇u_tau|^2 ∇u_tau)].
    """
    assert config.PROBLEM == 'MBE3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    ux, uy, uz = grad_fourier(u_tau, dx)
    s2 = ux*ux + uy*uy + uz*uz
    vx, vy, vz = s2*ux, s2*uy, s2*uz
    div_term   = div_fourier(vx, vy, vz, dx)
    lap_u      = laplacian_fourier_3d_phys(u_tau, dx)
    bi_u       = biharmonic(u_tau, dx)

    rhs_tau = lap_u - eps * bi_u + div_term

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    # H^{-1} metric
    B, S, _, _ = R.shape
    k2, inv_k2 = _fft_rk2(S, S, S, dx, R.device, dtype=torch.float32)
    R_hat = torch.fft.fftn(R.float(), dim=(1,2,3))
    r2 = (R_hat.real**2 + R_hat.imag**2) * inv_k2
    return torch.sqrt(r2 + 1e-8).mean().to(u_pred.dtype)

###################
# CH

# === ADD: CH3D RHS (dataset-consistent split) ================================
''''
def _rhs_ch3d(u, dx, eps):
    """
    CH3D gradient flow used in your MATLAB generator:
        u_t = - 2 Δu  -  ε^2 Δ^2 u  -  Δ (u^3 - 3u)
    Notes:
      • linear terms 2Δu + ε^2 Δ^2 u are the part you treated implicitly,
      • the cubic is explicit as Δ(u^3 - 3u),
      • 'eps' here is config.EPSILON_PARAM (ε), so we square it where needed.
    """
    lap_u   = laplacian_fourier_3d_phys(u, dx)      # Δu
    bi_u    = biharmonic(u, dx)                     # Δ^2 u
    chem    = u**3 - 3.0*u                          # f'(u) used by MATLAB
    lap_chem = laplacian_fourier_3d_phys(chem, dx)  # Δ f'(u)
    return -2.0 * lap_u - (eps**2) * bi_u - lap_chem
# ============================================================================
'''
# = ADD: CH3D semi-implicit teacher step (matches MATLAB exactly) ==========
'''''
def semi_implicit_step_ch(u_in, dt, dx, eps):
    """
    (1 + dt*(2k^2 + ε^2 k^4)) û^{n+1} = û^n - dt * k^2 * FFT( (u^n)^3 - 3 u^n )
    Returns real (B,S,S,S,1).
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)      # (2π/L)^2 |k|^2
    chem0_hat = torch.fft.fftn(u0**3 - 3.0*u0, dim=(1,2,3))
    U0        = torch.fft.fftn(u0, dim=(1,2,3))

    denom = 1.0 + dt * (2.0 * k2 + (eps**2) * (k2**2))
    numer = U0 - dt * k2 * chem0_hat
    U1 = numer / (denom + 1e-12)
    u1 = torch.fft.ifftn(U1, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)
'''''


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


def scheme_residual_fourier_ch(u_in, u_pred):
    """
    CORRECTED CH3D scheme residual to match MATLAB
    """
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)

    U0 = torch.fft.fftn(u0, dim=(1, 2, 3))
    UP = torch.fft.fftn(up, dim=(1, 2, 3))
    chem0_hat = torch.fft.fftn(u0 ** 3 - 3.0 * u0, dim=(1, 2, 3))

    denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))
    rhs = U0 - dt * k2 * chem0_hat

    rhat = denom * UP - rhs

    # Ignore k=0 for mass conservation (MATLAB does this implicitly)
    rhat = rhat * (k2 > 0)

    # Precondition
    rhat = rhat / (denom + 1e-12)

    r2 = rhat.real ** 2 + rhat.imag ** 2
    return r2.mean()


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
# ============================================================================


# === ADD: CH3D H^{-1} Gauss–Lobatto collocation residual ====================
import math as _m

def physics_collocation_tau_Hm1_CH(u_in, u_pred,
                                   tau=0.5 - 1.0/(2.0*_m.sqrt(5.0)),
                                   normalize=True):
    """
    CH3D collocation at u_tau, scored in H^{-1}:
      R_tau = (u^{n+1}-u^n)/dt - RHS_CH(u_tau)
    with optional per-sample normalization to balance ut vs RHS.
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()
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

    # H^{-1} metric (weight low-k, conserve mass dynamics)
    B, S, _, _ = R.shape
    k2, inv_k2 = _fft_rk2(S, S, S, dx, R.device, dtype=torch.float32)
    R_hat = torch.fft.fftn(R.float(), dim=(1,2,3))
    r2 = (R_hat.real**2 + R_hat.imag**2) * inv_k2
    return torch.sqrt(r2 + 1e-8).mean().to(u_pred.dtype)
# ============================================================================

# === NEW: CH3D Gauss–Lobatto collocation scored in L2 (identical form to SH/PFC/MBE) ===
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


def physics_guided_update_ch_optimal(u_in, y_pred, alpha_cap=0.6, low_k_snap_frac=0.45):
    """
    CH3D ONLY. Blend y_pred toward the CH semi-implicit teacher with optimal alpha*.
    We minimize the preconditioned semi-implicit residual in Fourier:

      (1 + dt*(2k^2 + eps^2 k^4)) û^{n+1} = û^n - dt * k^2 * FFT( (u^n)^3 - 3u^n )

    With:
      • light low-k snap to the teacher (improves large-scale coherence)
      • weighted LS alpha* in [0, alpha_cap]
      • monotone safety (don’t increase residual)
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0  = u_in.squeeze(-1).float()        # (B,S,S,S)
    up  = y_pred.squeeze(-1).float()
    usi = semi_implicit_step_ch(u_in, dt, dx, eps).squeeze(-1).float()

    B, S, _, _ = up.shape
    device = up.device
    k2 = _k_spectrum(S, S, S, dx, device)           # (2π/L)^2 |k|^2 >= 0
    k4 = k2 * k2

    U0   = torch.fft.fftn(u0,  dim=(1,2,3))
    UP   = torch.fft.fftn(up,  dim=(1,2,3))
    USI  = torch.fft.fftn(usi, dim=(1,2,3))
    CHEM0= torch.fft.fftn(u0**3 - 3.0*u0, dim=(1,2,3))

    denom = 1.0 + dt * (2.0 * k2 + (eps**2) * k4)
    rhs   = U0 - dt * k2 * CHEM0

    # ----- light low-k snap -----
    fx = torch.fft.fftfreq(S, d=1.0).to(device)
    FX, FY, FZ = torch.meshgrid(fx, fx, fx, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    mask_low = (r <= low_k_snap_frac * r.max()).to(UP.dtype)
    UP = mask_low * USI + (1.0 - mask_low) * UP

    # ----- preconditioned residual -----
    r0 = denom * UP  - rhs                 # residual at alpha=0
    a  = denom * (USI - UP)                # descent direction toward teacher
    w  = 1.0 / (denom + 1e-12)             # whitening

    def wdot(A, B): return (w * (A.real*B.real + A.imag*B.imag)).sum(dim=(1,2,3))

    with torch.no_grad():
        c1 = wdot(r0, a)
        c2 = (w * (a.real**2 + a.imag**2)).sum((1,2,3)) + 1e-24
        alpha = (-c1 / c2).clamp(0.0, alpha_cap).view(B,1,1,1)

        # monotone safety
        UP_star = UP + alpha * (USI - UP)
        rs = denom * UP_star - rhs
        r0_w = (w * (r0.real**2 + r0.imag**2)).sum((1,2,3))
        rs_w = (w * (rs.real**2 + rs.imag**2)).sum((1,2,3))
        alpha = torch.where((rs_w > r0_w).view(B,1,1,1),
                            torch.zeros_like(alpha), alpha)

    y_blend = up + alpha * (usi - up)
    # exact mass conservation afterwards (CH invariant)
    y_blend = mass_project_pred(y_blend.unsqueeze(-1), u_in).squeeze(-1)
    return y_blend.unsqueeze(-1).to(y_pred.dtype)


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


#################

#### AC3D

# ==== AC3D: Optimal Physics-Guided Update (semi-implicit residual LS) ====
def physics_guided_update_ac_optimal(u_in, y_pred, alpha_cap=0.6, low_k_snap_frac=0.45):
    """
    AC3D ONLY. Blend y_pred toward the AC semi-implicit teacher with optimal alpha*.
    Teacher (your AC semi-implicit step):
        û^{n+1} = [ û^n - (dt/eps^2) FFT(u^3 - u) ] / [ 1 + dt * k^2 ]
    We minimize the preconditioned semi-implicit residual in Fourier:
        R̂ = (1 + dt k^2) Û^{n+1} - [ Û^n - (dt/eps^2) FFT(u^3 - u) ]
    with:
      • light low-k spectral snap to teacher
      • weighted LS alpha* in [0, alpha_cap]
      • monotone safety (don’t increase residual)
    """
    assert config.PROBLEM == 'AC3D'
    dt, dx, eps2 = config.DT, config.DX, config.EPS2

    u0  = u_in.squeeze(-1).float()                 # (B,S,S,S)
    up  = y_pred.squeeze(-1).float()
    usi = semi_implicit_step(u_in, dt, dx, eps2).squeeze(-1).float()  # AC branch inside

    B, S, _, _ = up.shape
    device = up.device

    k2 = _k_spectrum(S, S, S, dx, device)         # (2π/L)^2 |k|^2 >= 0
    denom = 1.0 + dt * k2

    U0   = torch.fft.fftn(u0, dim=(1,2,3))
    UP   = torch.fft.fftn(up, dim=(1,2,3))
    USI  = torch.fft.fftn(usi, dim=(1,2,3))
    NL   = torch.fft.fftn(u0**3 - u0, dim=(1,2,3))  # dataset-consistent (no dealias for AC data)
    RHS  = U0 - (dt/eps2) * NL

    # light low-k snap to teacher
    fx = torch.fft.fftfreq(S, d=1.0).to(device)
    FX, FY, FZ = torch.meshgrid(fx, fx, fx, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    mask_low = (r <= low_k_snap_frac * r.max()).to(UP.dtype)
    UP = mask_low * USI + (1.0 - mask_low) * UP

    # preconditioned residual
    r0 = denom * UP  - RHS                    # residual at alpha=0
    a  = denom * (USI - UP)                   # descent direction
    w  = 1.0 / (denom + 1e-12)                # whitening

    def wdot(A, B): return (w * (A.real*B.real + A.imag*B.imag)).sum(dim=(1,2,3))

    with torch.no_grad():
        c1 = wdot(r0, a)
        c2 = (w * (a.real**2 + a.imag**2)).sum((1,2,3)) + 1e-24
        alpha = (-c1 / c2).clamp(0.0, alpha_cap).view(B,1,1,1)

        # monotone safety
        UP_star = UP + alpha * (USI - UP)
        rs = denom * UP_star - RHS
        r0_w = (w * (r0.real**2 + r0.imag**2)).sum((1,2,3))
        rs_w = (w * (rs.real**2 + rs.imag**2)).sum((1,2,3))
        alpha = torch.where((rs_w > r0_w).view(B,1,1,1), torch.zeros_like(alpha), alpha)

    y_blend = up + alpha * (usi - up)
    return y_blend.unsqueeze(-1).to(y_pred.dtype)
