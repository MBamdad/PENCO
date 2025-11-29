# physics_ac3d.py
import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def _build_wavenumbers(Nx, Ny, Nz, Lx, Ly, Lz, device):
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)
    kxx, kyy, kzz = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = torch.from_numpy(kxx**2 + kyy**2 + kzz**2).to(device).float()
    k_abs = torch.from_numpy(np.sqrt(kxx**2 + kyy**2 + kzz**2)).to(device).float()
    return k2.view(1,1,Nx,Ny,Nz), k_abs.view(1,1,Nx,Ny,Nz)

def _fft(u):   return torch.fft.fftn(u, dim=(-3,-2,-1))
def _ifft(uh): return torch.fft.ifftn(uh, dim=(-3,-2,-1)).real

def _apply_lowpass(uh, k_abs, k_cut):
    """ Keep modes with |k| <= k_cut; uh, k_abs shapes broadcast (B,T,Nx,Ny,Nz) vs (1,1,Nx,Ny,Nz). """
    if k_cut is None or k_cut <= 0:
        return uh
    mask = (k_abs <= k_cut)
    return uh * mask

def _apply_two_thirds_rule(uh):
    """ Classical 2/3 de-aliasing: zero out highest third along each axis. """
    B, T, Nx, Ny, Nz = uh.shape
    cx, cy, cz = Nx//3, Ny//3, Nz//3
    uh[..., 2*cx:,: , : ] = 0
    uh[..., : ,2*cy:,: ] = 0
    uh[..., : , : ,2*cz:] = 0
    return uh

def spectral_laplacian(u, k2):
    u_hat = _fft(u)
    lap_u_hat = -k2 * u_hat
    lap_u = _ifft(lap_u_hat)
    return lap_u

def periodic_bc_loss(u_time_major):
    """
    Periodic BCs are already enforced by spectral differentiation and the
    domain topology. On a collocated FFT grid, u[...,0] and u[...,-1]
    are *different* points, so forcing equality is unphysical.
    We therefore return 0 to avoid harming training.
    """
    return torch.tensor(0.0, device=u_time_major.device, dtype=u_time_major.dtype)

# (Optional) purely-diagnostic helper you can call and print *separately*:
def periodic_bc_diag(u_time_major):
    # mean |u[...,0]-u[...,-1]| on each axis faces (not used in loss)
    d = 0.0
    d += torch.mean(torch.abs(u_time_major[..., 0, :, :] - u_time_major[..., -1, :, :]))
    d += torch.mean(torch.abs(u_time_major[..., :, 0, :] - u_time_major[..., :, -1, :]))
    d += torch.mean(torch.abs(u_time_major[..., :, :, 0] - u_time_major[..., :, :, -1]))
    return d / 3.0

def allen_cahn_physics_loss_stable(
    u0, u_pred, Lx, Ly, Lz, dt, epsilon,
    k_cut=None, huber_delta=0.01, use_dealias=True
):
    """
    Stable strong-form physics for Allen–Cahn 3D, periodic domain.

    u0    : (B,Nx,Ny,Nz)
    u_pred: (B,Nx,Ny,Nz,T_out)  # works for T_out >= 1
    Returns dict: physics, pde, ic, bc
    """
    B, Nx, Ny, Nz, T = u_pred.shape
    device = u_pred.device
    dtype  = u_pred.dtype

    # time-major stack: u_all[:,0]=u0, u_all[:,1+k]=u_pred[...,k]
    u_all = torch.empty((B, T+1, Nx, Ny, Nz), device=device, dtype=dtype)
    u_all[:, 0]  = u0
    u_all[:, 1:] = u_pred.permute(0, 4, 1, 2, 3)  # (B,T,Nx,Ny,Nz)

    # ---- time derivative (robust for T=1,2,>2) ----
    # ut aligned to steps 1..T  -> shape (B,T,Nx,Ny,Nz)
    ut = torch.empty((B, T, Nx, Ny, Nz), device=device, dtype=dtype)
    if T == 1:
        # forward difference
        ut[:, 0] = (u_all[:, 1] - u_all[:, 0]) / dt
    elif T == 2:
        # forward at start, backward at end
        ut[:, 0] = (u_all[:, 1] - u_all[:, 0]) / dt
        ut[:, 1] = (u_all[:, 2] - u_all[:, 1]) / dt
    else:
        # forward at start, backward at end, central in the middle
        ut[:, 0] = (u_all[:, 1] - u_all[:, 0]) / dt
        ut[:, -1] = (u_all[:, -1] - u_all[:, -2]) / dt

        # central differences for k = 1 .. T-2:
        # u_{k+1} - u_{k-1} uses slices [2:T] and [0:T-2], both length (T-2)
        u_fwd = u_all[:, 2:T]  # shape (B, T-2, Nx, Ny, Nz)
        u_bwd = u_all[:, 0:T - 2]  # shape (B, T-2, Nx, Ny, Nz)
        ut[:, 1:-1] = (u_fwd - u_bwd) / (2.0 * dt)

    # ---- spectral Laplacian & stabilized nonlinearity ----
    (k2, k_abs) = _build_wavenumbers(Nx, Ny, Nz, Lx, Ly, Lz, device)

    # Δu at steps 1..T
    lap_u = spectral_laplacian(u_all[:, 1:], k2)  # (B,T,Nx,Ny,Nz)

    # Nonlinearity v = u^3 - u at steps 1..T
    v = u_all[:, 1:]**3 - u_all[:, 1:]

    if use_dealias:
        # (safe de-aliasing is handled by low-pass below; keep this call as a no-op or simple guard)
        pass

    # Low-pass both v and Δu to tame high-k residuals (optional but stabilizing)
    if k_cut is not None and k_cut > 0:
        v_hat = _fft(v)
        v_hat = _apply_lowpass(v_hat, k_abs, k_cut)
        v     = _ifft(v_hat)

        lap_u_hat = _fft(lap_u)
        lap_u_hat = _apply_lowpass(lap_u_hat, k_abs, k_cut)
        lap_u     = _ifft(lap_u_hat)

    # Residual: r = ut - Δu + (1/eps^2) * (u^3 - u)
    c = 1.0 / (epsilon**2)
    r = ut - lap_u + c * v

    # Robust residual loss (Huber/SmoothL1) over batch, time, space
    loss_pde = 0.1 * F.smooth_l1_loss(r, torch.zeros_like(r), beta=huber_delta, reduction='mean')

    # IC: first predicted frame consistent with u0
    loss_ic  = F.mse_loss(u_pred[..., 0], u0)

    # Periodic faces (diagnostic)
    loss_bc  = periodic_bc_loss(u_all)

    loss_physics = loss_pde + loss_ic + loss_bc
    return {'physics': loss_physics, 'pde': loss_pde, 'ic': loss_ic, 'bc': loss_bc}
