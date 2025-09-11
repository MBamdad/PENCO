import torch
import torch.nn.functional as F
import numpy as np
import config


# =========================
# FFT helpers / Laplacian
# =========================
@torch.no_grad()
def _fft_wavenumbers_3d(nx, ny, nz, dx):
    """
    Build -(2π)^2|k|^2 on a 3D mesh for the Laplacian in Fourier space.
    """
    kx = torch.fft.fftfreq(nx, d=dx)
    ky = torch.fft.fftfreq(ny, d=dx)
    kz = torch.fft.fftfreq(nz, d=dx)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    minus_k2 = -((2*np.pi)**2) * (kx**2 + ky**2 + kz**2)
    return minus_k2


def laplacian_fourier_3d_phys(u, dx):
    """
    ∇²u via FFT (periodic BCs). Compute in fp32/complex64 for AMP safety.
    u: (B,S,S,S) real tensor
    returns: (B,S,S,S) real tensor
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, nx, ny, nz = u.shape
        minus_k2 = _fft_wavenumbers_3d(nx, ny, nz, dx).to(u.device).to(torch.float32)
        u32 = u.float()
        u_ft = torch.fft.fftn(u32, dim=[1, 2, 3])          # complex64
        lap  = torch.fft.ifftn(minus_k2 * u_ft, dim=[1, 2, 3]).real  # real32
    return lap.to(u.dtype)


# =========================
# Optional scheme step (NOT used in trainer; kept for convenience)
# =========================
def semi_implicit_step(u_in, dt, dx, eps2):
    """
    One semi-implicit step for Allen–Cahn:
      (I - dt Δ) u^{n+1} = u^n - (dt/eps^2) ( (u^n)^3 - u^n )
    u_in: (B,S,S,S,1)
    returns: (B,S,S,S,1)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k = torch.fft.fftfreq(S, d=dx).to(u0.device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = (2*np.pi)**2 * (kx**2 + ky**2 + kz**2)

    nl = u0**3 - u0
    u0_hat = torch.fft.fftn(u0, dim=(1, 2, 3))
    nl_hat = torch.fft.fftn(nl, dim=(1, 2, 3))
    num = u0_hat - (dt/eps2) * nl_hat
    den = (1.0 + dt * k2)
    u1_hat = num / den
    u1 = torch.fft.ifftn(u1_hat, dim=(1, 2, 3)).real
    return u1.unsqueeze(-1)


# =========================
# Energetics
# =========================
def energy_density(u, dx, eps2):
    """
    Energy density: ε^2/2 |∇u|^2 + 1/4 (u^2 - 1)^2
    Computed via -0.5 ε^2 u Δu + 1/4 (u^2 - 1)^2
    """
    lap = laplacian_fourier_3d_phys(u, dx)
    grad2_term = -0.5 * eps2 * (u * lap)      # equals ε^2/2 |∇u|^2
    pot_term   = 0.25 * (u**2 - 1.0)**2
    return grad2_term + pot_term


def energy_total(u, dx, eps2):
    """Mean energy per sample."""
    return energy_density(u, dx, eps2).mean(dim=(1, 2, 3))


def energy_penalty(u_in, u_pred, dx, eps2):
    """
    Penalize energy increase: max(E^{n+1} - E^n, 0).
    """
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    E0 = energy_total(u0, dx, eps2)
    Ep = energy_total(up, dx, eps2)
    inc = torch.relu(Ep - E0)
    return inc.mean()


# =========================
# De-aliasing & μ(u)
# =========================
def dealias_two_thirds(u):
    """
    2/3-rule de-aliasing in RFFT space for cubic nonlinearity.
    u: (B,S,S,S)
    returns: (B,S,S,S)
    """
    S = u.shape[1]
    kcut = S // 3
    filt = torch.zeros((S, S, S//2 + 1), device=u.device, dtype=torch.float32)
    filt[:2*kcut, :2*kcut, :kcut+1] = 1.0
    uhat = torch.fft.rfftn(u, dim=(1, 2, 3))
    return torch.fft.irfftn(uhat * filt, s=(S, S, S), dim=(1, 2, 3)).real


def mu_ac(u, dx, eps2, dealias=True):
    """
    Chemical potential for Allen–Cahn:
      μ(u) = Δu - (1/ε^2) (u^3 - u)
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)
    if dealias:
        u = dealias_two_thirds(u)
    return lap_u - (1.0/eps2) * (u**3 - u)


def grad_mag_fft(u, dx):
    """
    |∇u| computed spectrally.
    """
    B, S, _, _ = u.shape
    k = torch.fft.fftfreq(S, d=dx).to(u.device)
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing='ij')
    KX = (1j * 2*np.pi) * KX
    KY = (1j * 2*np.pi) * KY
    KZ = (1j * 2*np.pi) * KZ
    U = torch.fft.fftn(u, dim=(1, 2, 3))
    ux = torch.fft.ifftn(KX * U, dim=(1, 2, 3)).real
    uy = torch.fft.ifftn(KY * U, dim=(1, 2, 3)).real
    uz = torch.fft.ifftn(KZ * U, dim=(1, 2, 3)).real
    return torch.sqrt(ux*ux + uy*uy + uz*uz + 1e-12)


# =========================
# Physics residuals (dimensionless/normalized)
# =========================
def physics_residual_midpoint(u_in, u_pred):
    """
    Strong midpoint residual:
      R = (u^{n+1}-u^n)/dt - μ( (u^{n+1}+u^n)/2 )
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    um = 0.5 * (u0 + up)
    ut = (up - u0) / dt
    mu_m = mu_ac(um, dx, eps2, dealias=True)
    Rm = ut - mu_m
    return (Rm**2).mean()


def physics_residual_random_collocation(u_in, u_pred, theta=None):
    """
    Strong residual at a random convex combination between n and n+1
    to reduce bias to endpoints.
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    if theta is None:
        theta = 0.3 + 0.4 * torch.rand(u0.shape[0], 1, 1, 1, device=u0.device)
    um = (1.0 - theta) * u0 + theta * up
    ut = (up - u0) / dt
    Rm = ut - mu_ac(um, dx, eps2, dealias=True)
    return (Rm**2).mean()


def physics_residual_weak_lowk(u_in, u_pred, kfrac=0.25):
    """
    Weak-form residual restricted to low-k band.
    IMPORTANT: Divide by S^3 because torch.fft.fftn is unnormalized.
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    ut = (up - u0) / dt
    R = ut - mu_ac(up, dx, eps2, dealias=True)

    B, S, _, _ = R.shape
    Rk = torch.fft.fftn(R, dim=(1, 2, 3))
    k = torch.fft.fftfreq(S, d=dx).to(R.device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    r = torch.sqrt(kx**2 + ky**2 + kz**2)
    mask = (r <= kfrac * r.max()).float()

    # Parseval fix: / S^3
    return (((Rk.real**2 + Rk.imag**2) * mask).mean() / (S**3))


def physics_residual_interface_weighted(u_in, u_pred, alpha=8.0, tau=None, q=0.75):
    """
    Interface-weighted strong residual using a sigmoid mask of |∇u|.
    tau is set adaptively per batch to the q-quantile of |∇u|.
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    ut = (up - u0) / dt
    R  = ut - mu_ac(up, dx, eps2, dealias=True)
    g  = grad_mag_fft(up, dx).detach()

    if tau is None:
        flat = g.flatten(1)
        k = int(max(1, round(q * flat.shape[1])))
        tau = torch.kthvalue(flat, k, dim=1).values.view(-1, 1, 1, 1)  # (B,1,1,1)

    w = torch.sigmoid(alpha * (g - tau))
    return ((w * R)**2).mean()


def energy_dissipation_identity_loss_midpoint(u_in, u_pred, huber_delta=1e-3):
    """
    Dimensionless midpoint EDI with Huber penalty:
      ((E_{n+1}-E_n)/|E_n|)/dt  + ε^2 * mean(μ(u_mid)^2)/(|E_n|+ε)  ≈ 0
    Returns: (loss, Ep_mean, E0_mean)
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    E0 = energy_total(u0, dx, eps2)                  # (B,)
    Ep = energy_total(up, dx, eps2)                  # (B,)
    um = 0.5 * (u0 + up)
    mu_m = mu_ac(um, dx, eps2, dealias=True)         # (B,S,S,S)

    denom = (E0.abs() + 1e-8)
    term_energy = ((Ep - E0) / denom) / dt
    term_mu2    = eps2 * (mu_m**2).mean(dim=(1, 2, 3)) / denom
    lhs = term_energy + term_mu2                     # (B,)

    # Huber penalty
    abs_lhs = lhs.abs()
    quad = 0.5 * (abs_lhs**2) / huber_delta
    lin  = abs_lhs - 0.5 * huber_delta
    loss = torch.where(abs_lhs <= huber_delta, quad, lin).mean()
    return loss, Ep.mean(), E0.mean()
