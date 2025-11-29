# post_processing.py

import torch
import torch.nn.functional as F
import numpy as np


def laplacian_fourier_3d(u, dx):
    """Computes 3D Laplacian using FFT. Assumes uniform grid spacing dx."""
    nx, ny, nz = u.shape[1], u.shape[2], u.shape[3]
    k_x = torch.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    k_y = torch.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
    k_z = torch.fft.fftfreq(nz, d=dx) * 2.0 * np.pi
    k_x, k_y, k_z = torch.meshgrid(k_x, k_y, k_z, indexing='ij')
    k_x, k_y, k_z = k_x.to(u.device), k_y.to(u.device), k_z.to(u.device)
    minus_k_squared = -(k_x ** 2 + k_y ** 2 + k_z ** 2)
    u_ft = torch.fft.fftn(u, dim=[1, 2, 3])
    u_lap_ft = minus_k_squared * u_ft
    u_lap = torch.fft.ifftn(u_lap_ft, dim=[1, 2, 3])
    return u_lap.real


def allen_cahn_pde_loss(u_in, u_traj_pred, grid_info):
    """
    Calculates the physics loss for the Allen-Cahn PDE over a predicted trajectory.
    Returns the total physics loss and debug values for its components.
    """
    S = u_in.shape[1]
    dx = 1.0 / S
    B, _, _, _, T_out, _ = u_traj_pred.shape

    # Unpack physics parameters
    epsilon = grid_info['epsilon']
    lambda_physics = grid_info['lambda_physics']
    dt = grid_info['dt_model']
    spatial_weight = grid_info['spatial_weight']

    full_traj = torch.cat((u_in.unsqueeze(-2), u_traj_pred), dim=-2)
    u_pred = full_traj[..., 1:, :].squeeze(-1)

    # --- Term 1: Time Derivative ---
    u_t_term = (full_traj[..., 1:, :] - full_traj[..., :-1, :]) / dt

    # --- Term 2: Spatial Component ---
    u_pred_reshaped = u_pred.permute(0, 4, 1, 2, 3).reshape(B * T_out, S, S, S)
    laplacian_u = laplacian_fourier_3d(u_pred_reshaped, dx)
    laplacian_u = laplacian_u.view(B, T_out, S, S, S).permute(0, 2, 3, 4, 1)

    reaction_term = u_pred ** 3 - u_pred
    mu_spatial = epsilon ** 2 * laplacian_u - reaction_term
    spatial_term = lambda_physics * mu_spatial

    # --- Form the balanced residual and calculate total physics loss ---
    residual = u_t_term.squeeze(-1) - spatial_weight * spatial_term
    pde_loss = F.mse_loss(residual, torch.zeros_like(residual))

    # --- For Debugging: Calculate magnitudes of the two balanced components ---
    # These are the two terms whose difference is minimized.
    debug_ut_mse = torch.mean(u_t_term ** 2)
    debug_spatial_mse = torch.mean((spatial_weight * spatial_term) ** 2)

    return pde_loss, debug_ut_mse, debug_spatial_mse