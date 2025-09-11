import torch
import torch.nn.functional as F
import numpy as np
import config

# -------------------------
# FFT helpers and Laplacian
# -------------------------
@torch.no_grad()
def _fft_wavenumbers_3d(nx, ny, nz, dx):
    # cycles per unit; Laplacian needs -(2π)^2|k|^2
    kx = torch.fft.fftfreq(nx, d=dx)
    ky = torch.fft.fftfreq(ny, d=dx)
    kz = torch.fft.fftfreq(nz, d=dx)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    minus_k2 = -((2*np.pi)**2) * (kx**2 + ky**2 + kz**2)
    return minus_k2

def laplacian_fourier_3d_phys(u, dx):
    """
    u: (B,S,S,S) real tensor; periodic BCs
    returns ∇²u via FFT (computed in fp32/complex64 for AMP safety)
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, nx, ny, nz = u.shape
        minus_k2 = _fft_wavenumbers_3d(nx, ny, nz, dx).to(u.device).to(torch.float32)  # real32
        u32 = u.float()
        u_ft = torch.fft.fftn(u32, dim=[1,2,3])  # complex64
        lap  = torch.fft.ifftn(minus_k2 * u_ft, dim=[1,2,3]).real  # real32
    return lap.to(u.dtype)

# -------------------------
# Allen–Cahn residuals (strong form)
# -------------------------
def physics_residual_matlab(u_in, u_pred):
    """
    Allen–Cahn PDE:
      u_t = Δu - (1/eps^2) * (u^3 - u)
    Residual at t+dt using u_pred:
      R = (u_pred - u_in)/dt - ( Δu_pred - (1/eps^2)(u_pred^3 - u_pred) )
    Returns:
      mse_phys, debug_ut_mse, debug_muspatial_mse (scaled)
    """
    dt  = config.DT
    dx  = config.DX
    eps2= config.EPS2

    u0 = u_in.squeeze(-1)   # (B,S,S,S)
    up = u_pred.squeeze(-1)

    ut = 1e0 * ((up - u0) / dt)
    lap_up = laplacian_fourier_3d_phys(up, dx)
    mu = 1e0 * (lap_up - (1.0/eps2)*(up**3 - up))

    R = ut - mu
    mse_phys = 1e0 * F.mse_loss(R, torch.zeros_like(R))

    debug_ut_mse = torch.mean(ut**2)
    debug_muspatial_mse = config.DEBUG_MU_SCALE * torch.mean(mu**2)
    return mse_phys, debug_ut_mse, debug_muspatial_mse

def physics_residual_normalized(u_in, u_pred):
    """
    Scale-balanced strong residual:
      R_tilde = ( (up-u0)/dt ) / RMS((up-u0)/dt)  -  μ(up) / RMS(μ(up))
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    ut = (up - u0) / dt
    lap_up = laplacian_fourier_3d_phys(up, dx)
    mu = lap_up - (1.0/eps2) * (up**3 - up)
    # per-batch RMS, detach to stop gradient
    s_t  = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    s_mu = mu.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    R_tilde = ut / s_t - mu / s_mu
    loss = R_tilde.pow(2).mean()
    return loss, s_t.mean(), s_mu.mean()

# -------------------------
# Semi-implicit one-step (teacher only — not used in decoder)
# -------------------------
def semi_implicit_step(u_in, dt, dx, eps2):
    # u_in: (B,S,S,S,1) real
    u0 = u_in.squeeze(-1).float()
    # FFTs
    B,S,_,_ = u0.shape[:4]
    kx = torch.fft.fftfreq(S, d=dx).to(u0.device)
    ky = torch.fft.fftfreq(S, d=dx).to(u0.device)
    kz = torch.fft.fftfreq(S, d=dx).to(u0.device)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k2 = (2*np.pi)**2 * (kx**2 + ky**2 + kz**2)  # >=0
    # Nonlinear term at time n (real space)
    nl = u0**3 - u0
    u0_hat = torch.fft.fftn(u0,   dim=[1,2,3])
    nl_hat = torch.fft.fftn(nl,   dim=[1,2,3])
    num = u0_hat - (dt/eps2) * nl_hat
    den = (1.0 + dt * k2)
    u1_hat = num / den
    u1 = torch.fft.ifftn(u1_hat, dim=[1,2,3]).real
    return u1.unsqueeze(-1)

# -------------------------
# Energy utilities
# -------------------------
def energy_density(u, dx, eps2):
    # u: (B,S,S,S) real
    lap = laplacian_fourier_3d_phys(u, dx)                    # (B,S,S,S)
    grad2_term = -0.5 * eps2 * (u * lap)                      # = eps^2/2 |∇u|^2
    pot_term   = 0.25 * (u**2 - 1.0)**2
    e = grad2_term + pot_term
    return e

def energy_penalty(u_in, u_pred, dx, eps2):
    """
    Only penalize energy *increase* from t^n to t^{n+1}.
    """
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    E0 = energy_density(u0, dx, eps2).mean(dim=(1,2,3))
    Ep = energy_density(up, dx, eps2).mean(dim=(1,2,3))
    inc = torch.relu(Ep - E0)           # only penalize increases
    return inc.mean()

# -------------------------
# Spectral dealiasing
# -------------------------
def dealias_two_thirds(u):
    # rfftn de-aliasing of cubic nonlinearity (2/3 rule)
    S = u.shape[1]
    kcut = S // 3
    filt = torch.zeros((S, S, S//2 + 1), device=u.device, dtype=torch.float32)
    filt[:2*kcut, :2*kcut, :kcut+1] = 1.0
    uhat = torch.fft.rfftn(u, dim=(1,2,3))
    return torch.fft.irfftn(uhat * filt, s=(S,S,S), dim=(1,2,3)).real

def mu_ac(u, dx, eps2, dealias=True):
    lap_u = laplacian_fourier_3d_phys(u, dx)
    if dealias:
        u = dealias_two_thirds(u)
    return lap_u - (1.0/eps2) * (u**3 - u)

# -------------------------
# Minimizing-movement (prox) and midpoint residual
# -------------------------
def mm_projection(u_in, u_pred, dt, dx, eps2, steps=5, eta=None):
    """
    Minimizing-movement proximal projection:
      u_{k+1} = u_k - eta * [ (u_k - u_in)/dt - mu(u_k) ].
    Returns refined u*, and the last gradient g*.
    """
    up = u_pred.squeeze(-1).float()
    u0 = u_in.squeeze(-1).float()
    if eta is None:
        eta = 0.5 * dt  # safe step size
    for _ in range(steps):
        g = (up - u0) / dt - mu_ac(up, dx, eps2, dealias=True)
        up = up - eta * g
    return up.unsqueeze(-1), g  # (B,S,S,S,1), (B,S,S,S)

def loss_mm_projection(u_in, u_pred):
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    up_ref, g_last = mm_projection(u_in, u_pred, dt, dx, eps2, steps=5)
    # two terms: projection distance and stationarity at the projected point
    l_proj = F.mse_loss(u_pred, up_ref)
    l_stat = (g_last**2).mean()
    return l_proj + l_stat, l_proj.detach(), l_stat.detach()

def physics_residual_midpoint(u_in, u_pred):
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    um = 0.5 * (u0 + up)
    ut = (up - u0) / dt
    mu_m = mu_ac(um, dx, eps2, dealias=True)
    Rm = ut - mu_m
    return (Rm**2).mean()

# -------------------------
# “Scheme residual” in Fourier (teacher consistency; not decoding)
# -------------------------
def scheme_residual_fourier(u_in, u_pred):
    """
    Compare LHS and RHS in Fourier for the semi-implicit discretization,
    but only as a residual loss (no scheme in decoder).
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    B,S,_,_ = u0.shape
    k = torch.fft.fftfreq(S, d=dx).to(u0.device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = (2*np.pi)**2 * (kx**2 + ky**2 + kz**2)

    # RHS (from u^n) with de-aliased cubic
    u0_da = dealias_two_thirds(u0)
    nl0_hat = torch.fft.fftn(u0_da**3 - u0_da, dim=(1,2,3))
    u0_hat  = torch.fft.fftn(u0, dim=(1,2,3))

    rhs_hat = u0_hat - (dt/eps2) * nl0_hat
    lhs_hat = (1.0 + dt * k2) * torch.fft.fftn(up, dim=(1,2,3))

    rhat = lhs_hat - rhs_hat
    return (rhat.real**2 + rhat.imag**2).mean()
