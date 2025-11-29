# trainer_hybrid.py

import torch
import torch.nn.functional as F
from timeit import default_timer
import numpy as np
from tqdm import tqdm
import time


# ====================================================================
# === PHYSICS-INFORMED COMPONENTS (THE CORE OF THE HYBRID MODEL) ===
# ====================================================================

def laplacian_fourier_3d(u, grid_info):
    """
    Calculates the 3D Laplacian using the Fourier method.
    Matches the MATLAB implementation for consistency.
    u: Tensor of shape (batch, Nx, Ny, Nz, T)
    """
    Nx, Ny, Nz = grid_info['Nx'], grid_info['Ny'], grid_info['Nz']
    Lx, Ly, Lz = grid_info['Lx'], grid_info['Ly'], grid_info['Lz']

    # Permute to (batch, T, Nx, Ny, Nz) for easier FFT
    u_perm = u.permute(0, 4, 1, 2, 3)

    # Get wave numbers
    kx = (2 * np.pi / Lx) * torch.fft.fftfreq(Nx, d=1.0 / Nx).to(u.device)
    ky = (2 * np.pi / Ly) * torch.fft.fftfreq(Ny, d=1.0 / Ny).to(u.device)
    kz = (2 * np.pi / Lz) * torch.fft.fftfreq(Nz, d=1.0 / Nz).to(u.device)
    kxx, kyy, kzz = torch.meshgrid(kx ** 2, ky ** 2, kz ** 2, indexing='ij')

    minus_k_sq = -(kxx + kyy + kzz).unsqueeze(0).unsqueeze(0)  # Add batch and time dims

    # Calculate Laplacian
    u_hat = torch.fft.fftn(u_perm, dim=[-3, -2, -1])
    u_lap_hat = minus_k_sq * u_hat
    u_lap = torch.fft.ifftn(u_lap_hat, dim=[-3, -2, -1]).real

    # Permute back to (batch, Nx, Ny, Nz, T)
    return u_lap.permute(0, 2, 3, 4, 1)


def allen_cahn_residual(u_pred, u_in, grid_info):
    """
    Calculates the residual of the Allen-Cahn PDE solved by the MATLAB code.
    PDE: u_t = ∇²u - (1/Cahn) * (u³ - u)

    u_pred: Predicted trajectory, shape (batch, Nx, Ny, Nz, T_out)
    u_in:   Input trajectory, shape (batch, Nx, Ny, Nz, T_in)
    """
    dt = grid_info['dt_model']
    Cahn = grid_info['Cahn']

    # Concatenate last input step with predicted steps for time derivative
    full_u = torch.cat([u_in[..., -1:], u_pred], dim=-1)

    # 1. Time derivative (u_t) using forward difference
    u_t = (full_u[..., 1:] - full_u[..., :-1]) / dt

    # 2. Laplacian (∇²u) on the predicted steps
    laplacian_u = laplacian_fourier_3d(u_pred, grid_info)

    # 3. Reaction term on the predicted steps
    reaction_term = (u_pred ** 3 - u_pred) / Cahn

    mu_spatial = laplacian_u - reaction_term

    # 4. PDE Residual
    residual = u_t - mu_spatial

    # For debugging and logging
    debug_ut_mse = 1e-1 * torch.mean(u_t ** 2)
    debug_mu_mse = torch.mean(mu_spatial ** 2)

    return residual, debug_ut_mse, debug_mu_mse


# ====================================================================
# === HYBRID TRAINING FUNCTION ===
# ====================================================================

def train_hybrid_fno4d(model, data_loss_fn, epochs, train_loader, test_loader,
                       optimizer, scheduler, normalized, normalizers, device,
                       pde_weight, grid_info):
    logs = {
        'total_loss': [], 'data_loss': [], 'pde_loss': [], 'test_l2': [],
        'u_t_mse': [], 'mu_spatial_mse': []
    }

    if normalized:
        normalizer_x, normalizer_y = normalizers[0], normalizers[1]

    for ep in range(epochs):
        model.train()
        t1 = time.time()

        # Accumulators for epoch averages
        avg_total_loss, avg_data_loss, avg_pde_res = 0.0, 0.0, 0.0
        avg_ut, avg_mu = 0.0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # --- Model Prediction ---
            y_pred_norm = model(x)

            # --- Un-normalize for physical losses ---
            if normalized:
                x_decoded = normalizer_x.decode(x)
                y_decoded = normalizer_y.decode(y)
                y_pred_decoded = normalizer_y.decode(y_pred_norm)
            else:
                x_decoded, y_decoded, y_pred_decoded = x, y, y_pred_norm

            # --- Loss Calculation ---
            # 1. Data Loss
            data_loss = data_loss_fn(y_pred_decoded.flatten(1), y_decoded.flatten(1))

            # 2. PDE (Physics) Loss
            residual, ut_mse, mu_mse = allen_cahn_residual(y_pred_decoded, x_decoded, grid_info)
            pde_loss =1e-4 *  F.mse_loss(residual, torch.zeros_like(residual))

            # 3. Total Hybrid Loss
            total_loss = data_loss * (1 - pde_weight) + pde_weight * pde_loss

            total_loss.backward()
            optimizer.step()

            # --- Accumulate Stats ---
            avg_total_loss += total_loss.item()
            avg_data_loss += data_loss.item()
            avg_pde_res += pde_loss.item()
            avg_ut += ut_mse.item()
            avg_mu += mu_mse.item()

        scheduler.step()

        # --- Epoch Evaluation ---
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if normalized:
                    out = normalizer_y.decode(out)
                    y = normalizer_y.decode(y)
                test_l2 += data_loss_fn(out.flatten(1), y.flatten(1)).item()

        # --- Average and Log ---
        num_batches = len(train_loader)
        avg_total_loss /= num_batches
        avg_data_loss /= num_batches
        avg_pde_res /= num_batches
        avg_ut /= num_batches
        avg_mu /= num_batches
        test_l2 /= len(test_loader.dataset)

        logs['total_loss'].append(avg_total_loss)
        logs['data_loss'].append(avg_data_loss)
        logs['pde_loss'].append(avg_pde_res)
        logs['test_l2'].append(test_l2)
        logs['u_t_mse'].append(avg_ut)
        logs['mu_spatial_mse'].append(avg_mu)

        t2 = time.time()

        # --- Print as requested ---
        if ep == 0:
            print("Epoch   Time(s)   Total Loss   Data Loss    PDE Res      Test L2      u_t Term MSE μ Term MSE")
        print(f"{ep:<7} {t2 - t1:<9.3f} {avg_total_loss:<12.4e} {avg_data_loss:<12.4e} {avg_pde_res:<12.4e} "
              f"{test_l2:<12.4e} {avg_ut:<12.4e} {avg_mu:<12.4e}")

    return model, logs


# ====================================================================
# === ORIGINAL DATA-DRIVEN TRAINING FUNCTION (for comparison) ===
# ====================================================================
def train_fno(model, data_loss_fn, epochs, train_loader, test_loader,
              optimizer, scheduler, normalized, normalizers, device):
    logs = {'data_loss': [], 'test_l2': []}
    y_normalizer = normalizers[1] if normalized else None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = data_loss_fn(out.flatten(1), y.flatten(1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                test_l2 += data_loss_fn(out.flatten(1), y.flatten(1)).item()

        train_l2 /= len(train_loader.dataset)
        test_l2 /= len(test_loader.dataset)
        logs['data_loss'].append(train_l2)
        logs['test_l2'].append(test_l2)
        t2 = default_timer()

        if ep == 0:
            print("Epoch   Time (s)    Train L2       Test L2")
        print(f"{ep:<7} {t2 - t1:<11.3f} {train_l2:<13.4e} {test_l2:<13.4e}")

    return model, logs