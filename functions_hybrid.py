# functions_hybrid.py

import torch
import torch.nn.functional as F
import numpy as np

def laplacian_fourier_3d(u, dx):
    """
    Calculates the 3D Laplacian using the Fourier method.
    Assumes periodic boundary conditions.
    u: Tensor of shape (batch, nx, ny, nz)
    dx: Grid spacing
    """
    nx, ny, nz = u.shape[1], u.shape[2], u.shape[3]
    k_x = torch.fft.fftfreq(nx, d=dx).to(u.device)
    k_y = torch.fft.fftfreq(ny, d=dx).to(u.device)
    k_z = torch.fft.fftfreq(nz, d=dx).to(u.device)
    kx, ky, kz = torch.meshgrid(k_x, k_y, k_z, indexing='ij')

    # The square of the wavevector magnitude, multiplied by (2*pi)^2
    minus_k_squared = -(kx**2 + ky**2 + kz**2) * (2 * np.pi)**2

    u_ft = torch.fft.fftn(u, dim=[1, 2, 3])
    u_lap_ft = minus_k_squared * u_ft
    u_lap = torch.fft.ifftn(u_lap_ft, dim=[1, 2, 3]).real
    return u_lap


def pde_loss_allen_cahn_3d(u_pred, grid_info):
    """
    Calculates the PDE loss for the Allen-Cahn equation for a predicted trajectory.

    *** MODIFIED TO EXACTLY MATCH THE MATLAB DATA GENERATION PDE ***
    MATLAB PDE:  ∂u/∂t = -∇²u - (1/ε²)(u³ - u)
    Residual form: R = ∂u/∂t + ∇²u + (1/ε²)(u³ - u) = 0
    """
    # Extract parameters from grid_info dictionary
    epsilon_param = grid_info['EPSILON_PARAM']
    dx = grid_info['Lx'] / grid_info['Nx']
    dt = grid_info['dt_model']

    batch_size, nx, ny, nz, T_out = u_pred.shape

    # 1. Calculate the time derivative (du/dt) using finite differences
    u_t_all = torch.diff(u_pred, dim=-1) / dt

    # To align dimensions, we'll evaluate the spatial part at the midpoint in time
    u_mid = (u_pred[..., :-1] + u_pred[..., 1:]) / 2.0

    # 2. Calculate the spatial terms for each time step
    pde_residuals = []

    # Pre-calculate the coefficient for the reaction term
    reaction_coeff = 1.0 / (epsilon_param ** 2)

    for t in range(T_out - 1):
        u_slice = u_mid[..., t]  # Shape: (batch, nx, ny, nz)

        # --- MATCHING MATLAB ---
        # MATLAB term is -∇²u, our laplacian function returns +∇²u
        # So we need to subtract it, which means adding -laplacian_u
        laplacian_u = laplacian_fourier_3d(u_slice, dx)

        # --- MATCHING MATLAB ---
        reaction_term = u_slice ** 3 - u_slice

        # --- ASSEMBLE THE RESIDUAL for this time step ---
        # R = u_t - (-laplacian_u - reaction_coeff * reaction_term)
        # R = u_t + laplacian_u + reaction_coeff * reaction_term
        residual = u_t_all[..., t] + laplacian_u + reaction_coeff * reaction_term
        pde_residuals.append(residual)

    # Stack residuals and compute the final loss
    all_residuals = torch.stack(pde_residuals, dim=-1)
    zeros_tensor = torch.zeros_like(all_residuals)
    total_loss = 1e-3 * F.mse_loss(all_residuals, zeros_tensor)

    # --- Diagnostics (Optional but good to keep) ---
    # We can calculate the MSE of the individual components against zero for logging
    # Note: These are not directly part of the loss, just for observation
    with torch.no_grad():
        u_t_mse = F.mse_loss(u_t_all, torch.zeros_like(u_t_all))

        # For mu, let's define it as the sum of the spatial terms
        mu_list = []
        for t in range(T_out - 1):
            u_slice = u_mid[..., t]
            lap = laplacian_fourier_3d(u_slice, dx)
            react = u_slice ** 3 - u_slice
            mu_term = -lap - reaction_coeff * react
            mu_list.append(mu_term)
        all_mu = torch.stack(mu_list, dim=-1)
        mu_mse = F.mse_loss(all_mu, torch.zeros_like(all_mu))

    return total_loss, u_t_mse, mu_mse