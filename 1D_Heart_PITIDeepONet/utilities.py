# utilities.py
import torch
import torch.nn.functional as F
from typing import Tuple

def to_dtype(t, dtype):
    return t.to(getattr(torch, dtype))

def relative_l2(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12):
    num = torch.linalg.norm(pred - ref)
    den = torch.linalg.norm(ref) + eps
    return (num / den).item()

def make_uniform_grid(x0: float, x1: float, Nx: int, device=None, dtype="float64"):
    x = torch.linspace(x0, x1, Nx, device=device, dtype=getattr(torch, dtype))
    return x

def periodic_rbf_kernel(x, y, length_scale, variance):
    """
    Periodic kernel: k(x,y) = variance * exp( - 2*sin^2(pi|x-y|)/l^2 )
    x: (Nx,), y: (Ny,), float64 recommended
    """
    X = x.unsqueeze(1)  # (Nx,1)
    Y = y.unsqueeze(0)  # (1,Ny)
    d = torch.pi * torch.abs(X - Y)
    sin2 = torch.sin(d) ** 2
    K = variance * torch.exp(-2.0 * sin2 / (length_scale ** 2))
    # Symmetrize to avoid tiny FP asymmetries
    return 0.5 * (K + K.T)

def _stable_cholesky(K, base_jitter, max_tries=8):
    I = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    jitter = base_jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + jitter * I)
        except RuntimeError:
            jitter *= 10.0
    # Fallback: eigen-decomposition
    w, V = torch.linalg.eigh(K)
    w_clamped = torch.clamp(w, min=1e-12)
    return V @ torch.diag(torch.sqrt(w_clamped))

def sample_periodic_gp_ic(x, length_scale, variance, seed=None, device=None, dtype="float64"):
    """
    Sample f ~ GP(0, k_periodic). Enforce Dirichlet endpoints: f(0)=f(1)=0.
    Robust to PD issues via adaptive jitter + (fallback) eigensampling. Uses float64.
    """
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    x = x.to(getattr(torch, dtype))
    K = periodic_rbf_kernel(x, x, length_scale, variance).to(getattr(torch, dtype))
    # Base jitter scaled by variance & grid size (more stable for large σ²)
    base_jitter = 1e-9 * max(1.0, float(variance))  # adaptive
    L = _stable_cholesky(K, base_jitter=base_jitter)
    z = torch.randn(len(x), generator=g, device=x.device, dtype=x.dtype)
    f = L @ z
    # Enforce Dirichlet endpoints
    f[0] = 0.0
    f[-1] = 0.0
    return f

def second_derivative_1d_interior(u, dx):
    """
    2nd-order central differences for interior nodes (vectorized).
    """
    Nx = u.shape[0]
    u_xx = torch.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    return u_xx

def second_derivative_1d_onesided_4th(u, dx):
    """
    4th-order one-sided 2nd-derivative at boundaries.
    Left (i=0) and Right (i=N-1) mirrors.
    """
    Nx = u.shape[0]
    u_xx = second_derivative_1d_interior(u, dx)
    if Nx < 6:
        raise ValueError("Need Nx>=6 for 4th-order one-sided boundary stencils.")
    coeff = torch.tensor([45, -154, 214, -156, 61, -10],
                         device=u.device, dtype=u.dtype) / (12.0 * dx * dx)
    u_xx[0] = torch.dot(coeff, u[0:6])
    coeff_r = torch.flip(coeff, dims=[0])
    u_xx[-1] = torch.dot(coeff_r, u[-6:])
    return u_xx

def explicit_heat_step(u, alpha, dx, dt):
    """
    Explicit Euler step for u_t = alpha u_xx with mixed stencils.
    """
    u_xx = second_derivative_1d_onesided_4th(u, dx)
    unew = u + dt * alpha * u_xx
    unew[0] = 0.0
    unew[-1] = 0.0
    return unew

def simulate_heat_1d(u0, alpha, x, Nt, dt):
    """
    Produce U(t_k, x) and Ut(t_k, x). Shapes: (Nt+1, Nx).
    """
    Nx = x.shape[0]
    dx = (x[-1] - x[0]) / (Nx - 1)
    U = torch.zeros(Nt + 1, Nx, device=u0.device, dtype=u0.dtype)
    Ut = torch.zeros_like(U)
    u = u0.clone()
    U[0] = u
    u_xx = second_derivative_1d_onesided_4th(u, dx)
    Ut[0] = alpha * u_xx
    for k in range(Nt):
        u = explicit_heat_step(u, alpha, dx, dt)
        U[k + 1] = u
        u_xx = second_derivative_1d_onesided_4th(u, dx)
        Ut[k + 1] = alpha * u_xx
    return U, Ut

def downsample_linear_time(U_fine, Ut_fine, T_final, Nt_coarse):
    """
    Linear interpolation over time to Nt_coarse+1 samples in [0, T_final].
    """
    Nt_fine = U_fine.shape[0] - 1
    t_fine = torch.linspace(0.0, T_final, Nt_fine + 1, device=U_fine.device, dtype=U_fine.dtype)
    t_coarse = torch.linspace(0.0, T_final, Nt_coarse + 1, device=U_fine.device, dtype=U_fine.dtype)

    def interp(A):
        Ac = torch.zeros(Nt_coarse + 1, A.shape[1], device=A.device, dtype=A.dtype)
        idx_float = (t_coarse / T_final) * Nt_fine
        idx0 = torch.clamp(idx_float.floor().long(), 0, Nt_fine - 1)
        idx1 = idx0 + 1
        w = (t_coarse - t_fine[idx0]) / (t_fine[idx1] - t_fine[idx0] + 1e-20)
        w = w.unsqueeze(1)
        Ac = (1 - w) * A[idx0] + w * A[idx1]
        return Ac

    Uc = interp(U_fine)
    Utc = interp(Ut_fine)
    return Uc, Utc

# --------- Time integration for learned operator G_theta ---------

@torch.no_grad()
def euler_step(G, u_n, dt, x, device):
    ut = G(u_n, x)
    return u_n + dt * ut

@torch.no_grad()
def rk4_step(G, u_n, dt, x, device):
    k1 = G(u_n, x)
    k2 = G(u_n + 0.5 * dt * k1, x)
    k3 = G(u_n + 0.5 * dt * k2, x)
    k4 = G(u_n + dt * k3, x)
    return u_n + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@torch.no_grad()
def abm2_step(G, u_nm1, u_n, dt, x, device):
    ut_n = G(u_n, x)
    ut_nm1 = G(u_nm1, x)
    u_pred = u_n + dt * (1.5 * ut_n - 0.5 * ut_nm1)
    ut_pred = G(u_pred, x)
    u_np1 = u_n + 0.5 * dt * (ut_n + ut_pred)
    return u_np1

def make_integrator(name: str):
    name = name.lower()
    if name == "euler":
        return euler_step
    if name == "rk4":
        return rk4_step
    if name == "abm2":
        return abm2_step
    raise ValueError(f"Unknown integrator {name}")

def residual_pointwise(u_in, u_hat):
    return (u_in - u_hat) ** 2
