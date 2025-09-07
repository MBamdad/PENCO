import torch
import torch.nn.functional as F
from timeit import default_timer
import numpy as np
from tqdm import tqdm
import time


# Original train fno
def train_fno(model, myloss, epochs, batch_size, train_loader, test_loader,
              optimizer, scheduler, normalized, normalizer, device):
    train_mse_log = []
    train_l2_log = []
    test_l2_log = []

    if normalized:
        # a_normalizer = normalizer[0].to(device)
        y_normalizer = normalizer[1].to(device)
    else:
        # a_normalizer = None
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            #print("x shape", x.shape)
            out = model(x)
            #print(f"Input shape: {x.shape}, y (target): {y.shape}, prediction (model output): {out.shape}")
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            mse = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
            loss = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += loss.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                #test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
                test_l2 += myloss(out.flatten(start_dim=1), y.flatten(start_dim=1)).item()

        train_mse /= len(train_loader)
        train_l2 /= (batch_size * len(train_loader))
        test_l2 /= (batch_size * len(test_loader))

        train_mse_log.append(train_mse)
        train_l2_log.append(train_l2)
        test_l2_log.append(test_l2)

        # Update the learning rate based on the test_l2 metric
        #scheduler.step(test_l2) ##


        t2 = default_timer()
        #print(ep, t2 - t1, train_mse, train_l2, test_l2)
        if ep == 0:  # Print the header row once
            print("No. Epoch   Time (s)       Train MSE      Train L2       Test L2")

        print(f"{ep:<10} {t2 - t1:<13.6f} {train_mse:<13.10f} {train_l2:<13.10f} {test_l2:<13.10f}")

    return model, train_mse_log, train_l2_log, test_l2_log

'''

def train_fno(model, myloss, adam_epochs, lbfgs_epochs, batch_size, train_loader, test_loader,
              optimizer_adam, scheduler, normalized, normalizer, device):
    """
    A comprehensive FNO training function with a two-phase optimization: Adam followed by L-BFGS.
    """
    # Initialize logs for the entire training process
    train_mse_log = []
    train_l2_log = []
    test_l2_log = []

    # Ensure normalizer is on the correct device
    if normalized and normalizer:
        if normalizer[1] is not None:
            normalizer[1].to(device)

    # =========================================================
    # =========== PHASE 1: ADAM OPTIMIZATION ==================
    # =========================================================
    print(f"\n--- [Data-Only] Phase 1: Adam optimization for {adam_epochs} epochs ---")
    for ep in range(adam_epochs):
        model.train()
        t1 = default_timer()
        epoch_train_mse = 0
        epoch_train_l2 = 0
        for x, y in tqdm(train_loader, desc=f"Adam Epoch {ep + 1}/{adam_epochs} [Training]"):
            x, y = x.to(device), y.to(device)
            optimizer_adam.zero_grad()

            l2_loss, mse_loss = _compute_fno_loss(model, x, y, myloss, normalized, normalizer, device)

            l2_loss.backward()
            optimizer_adam.step()
            scheduler.step()

            epoch_train_mse += mse_loss.item()
            epoch_train_l2 += l2_loss.item()

        # Log training losses, matching the original function's scaling
        train_mse_log.append(epoch_train_mse / len(train_loader))
        train_l2_log.append(epoch_train_l2 / (batch_size * len(train_loader)))

        # Validation loop
        model.eval()
        epoch_test_l2 = 0.0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Adam Epoch {ep + 1}/{adam_epochs} [Testing]"):
                x, y = x.to(device), y.to(device)
                l2_loss, _ = _compute_fno_loss(model, x, y, myloss, normalized, normalizer, device)
                epoch_test_l2 += l2_loss.item()

        test_l2_log.append(epoch_test_l2 / (batch_size * len(test_loader)))
        t2 = default_timer()

        if ep == 0:
            print("No. Epoch   Time (s)       Train MSE      Train L2       Test L2")
        print(
            f"{ep:<10} {t2 - t1:<13.6f} {train_mse_log[-1]:<13.10f} {train_l2_log[-1]:<13.10f} {test_l2_log[-1]:<13.10f}")
        print("-" * 80)

    # =========================================================
    # ========== PHASE 2: L-BFGS OPTIMIZATION =================
    # =========================================================
    if lbfgs_epochs > 0:
        print(f"\n--- [Data-Only] Phase 2: L-BFGS optimization for {lbfgs_epochs} epochs ---")
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, history_size=100,
                                            line_search_fn='strong_wolfe')

        def closure():
            optimizer_lbfgs.zero_grad()
            loss_agg = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                l2_loss, _ = _compute_fno_loss(model, x, y, myloss, normalized, normalizer, device)
                (l2_loss / len(train_loader)).backward()
                loss_agg += l2_loss.item()
            return loss_agg / len(train_loader)

        for ep in range(lbfgs_epochs):
            model.train()
            t1 = default_timer()
            optimizer_lbfgs.step(closure)

            model.eval()
            with torch.no_grad():
                # Re-evaluate and log losses on full train and test sets
                epoch_train_mse, epoch_train_l2, epoch_test_l2 = 0, 0, 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    l2_loss, mse_loss = _compute_fno_loss(model, x, y, myloss, normalized, normalizer, device)
                    epoch_train_mse += mse_loss.item()
                    epoch_train_l2 += l2_loss.item()
                train_mse_log.append(epoch_train_mse / len(train_loader))
                train_l2_log.append(epoch_train_l2 / (batch_size * len(train_loader)))

                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    l2_loss, _ = _compute_fno_loss(model, x, y, myloss, normalized, normalizer, device)
                    epoch_test_l2 += l2_loss.item()
                test_l2_log.append(epoch_test_l2 / (batch_size * len(test_loader)))

            t2 = default_timer()
            print(
                f"{adam_epochs + ep:<10} {t2 - t1:<13.6f} {train_mse_log[-1]:<13.10f} {train_l2_log[-1]:<13.10f} {test_l2_log[-1]:<13.10f}")
            print("-" * 80)

    return model, train_mse_log, train_l2_log, test_l2_log
'''
######
####
def train_fno_time(model, myloss, epochs, batch_size, train_loader, test_loader,
                   optimizer, scheduler, normalized, normalizer, device):
    ntrain = len(train_loader) * train_loader.batch_size
    ntest = len(test_loader) * test_loader.batch_size
    train_mse_log = []
    train_l2_log = []
    test_l2_log = []
    step = 1
    if normalized:
        a_normalizer = normalizer[0].to(device)
        y_normalizer = normalizer[1].to(device)
    else:
        a_normalizer = None
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            T = yy.shape[-1]
            #print(f" T : {T}")
            #print(f"target shape: {yy.shape}")
            #print(f"Input shape: {xx.shape}, y (target): {yy.shape}")
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                #print(f"Input shape: {xx.shape}, y (target): {y.shape}, prediction (model output): {im.shape}")
                #print(f"target shape 2: {yy.shape}")
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)

            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        t2 = default_timer()
        train_mse = train_l2_step / ntrain / (T / step)
        train_l2 = train_l2_full / ntrain
        test_l2 = test_l2_full / ntest

        # Log the loss values
        train_l2_log.append(train_l2_step / ntrain / (T / step))
        test_l2_log.append(test_l2_step / ntest / (T / step))

        if ep == 0:  # Print the header row once
            print("No. Epoch   Time (s)       Train MSE      Train L2       Test L2")

        print(f"{ep:<10} {t2 - t1:<13.6f} {train_mse:<13.10f} {train_l2:<13.10f} {test_l2:<13.10f}")

    return model, train_l2_log, test_l2_log

'''
## Use two Optimizers
# A helper function for the data-only loss to keep the main loop clean
def _compute_fno_loss(model, x, y, myloss, normalized, normalizer, device):
    """
    Computes the data-driven loss for a single batch for the FNO model.
    Returns the loss for backprop (LpLoss) and the loss for logging (MSE).
    """
    out = model(x)

    if normalized and normalizer:
        # Assumes normalizer is a list/tuple, and we need the second element for y
        y_normalizer = normalizer[1]
        out_decoded = y_normalizer.decode(out)
        y_decoded = y_normalizer.decode(y)
    else:
        out_decoded = out
        y_decoded = y

    # Loss for backpropagation (as in the original function)
    l2_loss = myloss(out_decoded.flatten(start_dim=1), y_decoded.flatten(start_dim=1))

    # MSE loss for logging (as in the original function)
    mse_loss = F.mse_loss(out_decoded.flatten(start_dim=1), y_decoded.flatten(start_dim=1), reduction='mean')

    return l2_loss, mse_loss

########################
########################
# Place this function near pde_residual
def loss_boundary(u_phys):
    """
    Computes the loss for periodic boundary conditions.
    Assumes u_phys is a tensor of shape [batch, Nx, Ny, Nz, T_total].
    """
    # Periodicity along the x-axis (dim=1)
    bc_x_loss = torch.mean((u_phys[:, 0, :, :, :] - u_phys[:, -1, :, :, :]) ** 2)

    # Periodicity along the y-axis (dim=2)
    bc_y_loss = torch.mean((u_phys[:, :, 0, :, :] - u_phys[:, :, -1, :, :]) ** 2)

    # Periodicity along the z-axis (dim=3)
    bc_z_loss = torch.mean((u_phys[:, :, :, 0, :] - u_phys[:, :, :, -1, :]) ** 2)

    return bc_x_loss + bc_y_loss + bc_z_loss

# A helper function to avoid repeating the loss calculation logic
# MODIFIED: Added bc_weight and calculation for boundary_loss
# MODIFIED: A more flexible helper function for combining losses
def _compute_hybrid_loss(model, x, y, myloss, normalizers, grid_info,
                         pde_weight, bc_weight, pde_loss_scaler, epsilon, device):
    """Computes the hybrid loss for a single batch."""
    normalized = (normalizers is not None)
    out = model(x)
    data_loss = myloss(out.view(x.size(0), -1), y.view(x.size(0), -1))

    pde_loss, boundary_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    if pde_weight > 0 or bc_weight > 0:
        # ... (decoding logic as before) ...
        if normalized:
            # ...
            x_phys = normalizers[0].decode(x)
            out_phys = normalizers[1].decode(out)
        else:
            x_phys, out_phys = x, out
        u_full_phys = torch.cat((x_phys, out_phys), dim=-1)

        if pde_weight > 0:
            residual = pde_residual(u_full_phys, grid_info, epsilon)
            pde_loss = torch.mean(residual ** 2)

        if bc_weight > 0:
            boundary_loss = loss_boundary(u_full_phys)

    scaled_pde_loss = pde_loss_scaler * pde_loss

    # A more flexible way to combine losses, weights act as multipliers
    total_loss = data_loss + pde_weight * scaled_pde_loss + bc_weight * boundary_loss

    return total_loss, data_loss, scaled_pde_loss, pde_loss, boundary_loss
'''

''''
def calculate_pde_residual(u_phys, grid_info, epsilon, problem, device):
    """
    Calculates the residual for a specified 3D PDE on PHYSICAL data.
    u_phys shape: (batch, Nx, Ny, Nz, T_out) - Denormalized data
    grid_info: Dictionary containing Nx, Ny, Nz, Lx, Ly, Lz, dt_model, T_out
    pde_params: Dictionary containing PDE-specific parameters
    problem_name: String identifier for the PDE (e.g., 'SH3D', 'AC3D', 'CH3D', 'MBE3D', 'PFC3D')
    device: PyTorch device
    """
    batch_size, Nx, Ny, Nz, T_out = u_phys.shape
    Lx, Ly, Lz = grid_info['Lx'], grid_info['Ly'], grid_info['Lz']
    dt_model = grid_info['dt_model']

    if T_out <= 1:
        print(f"Warning: T_out ({T_out}) <= 1 for problem {problem}. PDE loss requires T_out > 1. Returning zero loss.")
        return torch.zeros(1, device=device, requires_grad=True)

    # --- Calculate Time Derivative (∂u/∂t) ---
    du_dt = torch.zeros_like(u_phys)
    du_dt[..., 0] = (u_phys[..., 1] - u_phys[..., 0]) / dt_model
    du_dt[..., -1] = (u_phys[..., -1] - u_phys[..., -2]) / dt_model
    if T_out > 2:
       du_dt[..., 1:-1] = (u_phys[..., 2:] - u_phys[..., :-2]) / (2 * dt_model)

    # --- Common Spectral Derivative Setup ---
    _kx = torch.fft.fftfreq(Nx, d=Lx/Nx) * 2 * torch.pi
    _ky = torch.fft.fftfreq(Ny, d=Ly/Ny) * 2 * torch.pi
    _kz = torch.fft.fftfreq(Nz, d=Lz/Nz) * 2 * torch.pi

    ikx_m, iky_m, ikz_m = torch.meshgrid(1j * _kx, 1j * _ky, 1j * _kz, indexing='ij')
    ikx_m = ikx_m.to(device)
    iky_m = iky_m.to(device)
    ikz_m = ikz_m.to(device)

    k2x_m, k2y_m, k2z_m = torch.meshgrid(_kx**2, _ky**2, _kz**2, indexing='ij')
    # k2_m is kx^2 + ky^2 + kz^2. In Fourier space, laplacian is -k2_m
    k2_m = (k2x_m + k2y_m + k2z_m).to(device)

    u_hat = torch.fft.fftn(u_phys, dim=[1, 2, 3])

    pde_residual = None

    if problem == 'SH3D':
        epsilon_sh = epsilon # pde_params.get('epsilon_sh')
        if epsilon_sh is None: raise ValueError("Parameter 'epsilon_sh' not provided for SH3D.")
        k4_m = k2_m**2
        lap_u_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * u_hat
        biharm_u_hat = k4_m.unsqueeze(0).unsqueeze(-1) * u_hat
        lap_u = torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]).real
        biharm_u = torch.fft.ifftn(biharm_u_hat, dim=[1, 2, 3]).real
        rhs_sh3d = -(u_phys**3) - (1 - epsilon_sh) * u_phys - biharm_u - 2 * lap_u
        pde_residual = du_dt - rhs_sh3d

    elif problem == 'AC3D':
        Cahn_ac = epsilon # pde_params.get('Cahn_ac')
        if Cahn_ac is None: raise ValueError("Parameter 'Cahn_ac' not provided for AC3D.")
        lap_u_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * u_hat
        lap_u = torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]).real
        f_prime_u = u_phys**3 - u_phys
        rhs_ac3d = lap_u - (1/Cahn_ac) * f_prime_u
        pde_residual = du_dt - rhs_ac3d

    elif problem == 'CH3D':
        Cahn_ch = epsilon #  pde_params.get('Cahn_ch')
        if Cahn_ch is None: raise ValueError("Parameter 'Cahn_ch' not provided for CH3D.")
        lap_u_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * u_hat
        lap_u = torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]).real
        #mu_terms = (u_phys**3 - 3 * u_phys) - Cahn_ch * lap_u
        mu_terms = (u_phys ** 3 - 3 * u_phys) - Cahn_ch * lap_u
        mu_terms_hat = torch.fft.fftn(mu_terms, dim=[1, 2, 3])
        lap_mu_terms_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * mu_terms_hat
        lap_mu_terms = torch.fft.ifftn(lap_mu_terms_hat, dim=[1, 2, 3]).real
        rhs_ch3d = lap_mu_terms
        pde_residual = du_dt - rhs_ch3d

    elif problem == 'MBE3D':
        epsilon_mbe = epsilon #  pde_params.get('epsilon_mbe')
        if epsilon_mbe is None: raise ValueError("Parameter 'epsilon_mbe' not provided for MBE3D.")
        du_dx_hat = ikx_m.unsqueeze(0).unsqueeze(-1) * u_hat
        du_dy_hat = iky_m.unsqueeze(0).unsqueeze(-1) * u_hat
        du_dz_hat = ikz_m.unsqueeze(0).unsqueeze(-1) * u_hat
        du_dx = torch.fft.ifftn(du_dx_hat, dim=[1, 2, 3]).real
        du_dy = torch.fft.ifftn(du_dy_hat, dim=[1, 2, 3]).real
        du_dz = torch.fft.ifftn(du_dz_hat, dim=[1, 2, 3]).real
        grad_u_sq = du_dx**2 + du_dy**2 + du_dz**2
        f1 = grad_u_sq * du_dx
        f2 = grad_u_sq * du_dy
        f3 = grad_u_sq * du_dz
        f1_hat = torch.fft.fftn(f1, dim=[1, 2, 3])
        f2_hat = torch.fft.fftn(f2, dim=[1, 2, 3])
        f3_hat = torch.fft.fftn(f3, dim=[1, 2, 3])
        div_term_hat = (ikx_m.unsqueeze(0).unsqueeze(-1) * f1_hat +
                        iky_m.unsqueeze(0).unsqueeze(-1) * f2_hat +
                        ikz_m.unsqueeze(0).unsqueeze(-1) * f3_hat)
        div_term = torch.fft.ifftn(div_term_hat, dim=[1, 2, 3]).real
        lap_u_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * u_hat
        lap_u = torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]).real
        k4_m = k2_m**2
        biharm_u_hat = k4_m.unsqueeze(0).unsqueeze(-1) * u_hat
        biharm_u = torch.fft.ifftn(biharm_u_hat, dim=[1, 2, 3]).real
        rhs_mbe3d = -lap_u - epsilon_mbe * biharm_u - div_term
        pde_residual = du_dt - rhs_mbe3d

    elif problem == 'PFC3D':
        epsilon_pfc = epsilon # pde_params.get('epsilon_pfc') # Parameter 'r' in PFC, often -(epsilon)
        if epsilon_pfc is None: raise ValueError("Parameter 'epsilon_pfc' not provided for PFC3D.")

        # Calculate necessary derivatives
        # -∇²u  (term1_spatial_operator * u)
        lap_u_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * u_hat
        lap_u = torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]).real

        # ∇⁴u   (term2_spatial_operator * u)
        k4_m = k2_m**2
        biharm_u_hat = k4_m.unsqueeze(0).unsqueeze(-1) * u_hat
        biharm_u = torch.fft.ifftn(biharm_u_hat, dim=[1, 2, 3]).real

        # -∇⁶u  (term3_spatial_operator * u)
        k6_m = k2_m**3 # Be careful with sign if k2_m is just kx^2+ky^2+kz^2
                      # Since k^6 in Fourier space corresponds to (-∇^2)^3 = -∇^6 if direct multiplication
                      # Or (i k)^6 = -k^6.
                      # The MATLAB denominator (1/dt + (1-eps)k^2 + k^6) implies the k^6 term is positive.
                      # So in real space, this corresponds to -∇⁶u (because -(-∇²)^3 u = ∇⁶u is not what we want)
                      # A positive k^6 in Fourier space means we multiply by k^6, which is (-(laplacian_operator))^3,
                      # this will become -laplacian_operator^3 which is -∇⁶.
                      # The MATLAB form `(pp2+qq2+rr2).^3` is `(k^2)^3 = k^6`.
                      # So this term becomes `k^6 * u_hat` -> `∇⁶u` (with a negative sign from implicit to explicit)
        # Or, if (pp2+qq2+rr2) is k^2, then (pp2+qq2+rr2)^3 is k^6.
        # The term in the denominator is `+ k^6`, so when moved to RHS it's `-k^6 u_hat`.
        # `-k^6 u_hat` corresponds to `∇⁶u` in real space.
        triharm_u_hat = -k6_m.unsqueeze(0).unsqueeze(-1) * u_hat
        triharm_u = torch.fft.ifftn(triharm_u_hat, dim=[1, 2, 3]).real


        # ∇²(u³)
        u_cubed = u_phys**3
        u_cubed_hat = torch.fft.fftn(u_cubed, dim=[1, 2, 3])
        lap_u_cubed_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * u_cubed_hat
        lap_u_cubed = torch.fft.ifftn(lap_u_cubed_hat, dim=[1, 2, 3]).real

        # PDE: ∂u/∂t + (1-ε)∇²u + 2∇⁴u + ∇⁶u + ∇²(u³) = 0
        # RHS = - ( (1-ε_pfc)∇²u + 2∇⁴u + ∇⁶u + ∇²(u³) )
        # Let's use the parameter `epsilon` from PFC MATLAB as `r` (undercooling param)
        # A common PFC form is du/dt = nabla^2 * ( (r)u + u^3 - (1+nabla^2)^2 u )
        # du/dt = nabla^2 * ( ru + u^3 - (1 + 2 nabla^2 + nabla^4)u )
        # du/dt = r nabla^2 u + nabla^2(u^3) - nabla^2 u - 2 nabla^4 u - nabla^6 u
        # du/dt = (r-1) nabla^2 u - 2 nabla^4 u - nabla^6 u + nabla^2(u^3)
        # Here 'r' from literature is often called 'epsilon' in PFC code.
        # Let's match the form derived from MATLAB:
        # ∂u/∂t = -(1-ε)∇²u - 2∇⁴u - ∇⁶u - ∇²(u³)
        # Residual: du_dt - [-(1-epsilon_pfc)*lap_u - 2*biharm_u - triharm_u - lap_u_cubed]

        term1_spatial = (1 - epsilon_pfc) * lap_u # (1-eps)(-nabla^2 u) -> (eps-1)nabla^2 u
        term2_spatial = 2 * biharm_u             # 2 nabla^4 u
        term3_spatial = triharm_u                # nabla^6 u
        term4_nonlinear = lap_u_cubed            # nabla^2 (u^3)

        # According to the derived form: ∂u/∂t = -(1-ε)∇²u - 2∇⁴u - ∇⁶u - ∇²(u³)
        # residual = du_dt - (-(1-epsilon_pfc)*lap_u - 2*biharm_u - triharm_u - lap_u_cubed)
        rhs_pfc3d = (1 - epsilon_pfc) * lap_u + 2 * biharm_u + triharm_u + lap_u_cubed
        pde_residual = du_dt - rhs_pfc3d

    else:
        raise ValueError(f"Unknown problem_name: {problem}. PDE residual not defined.")

    if pde_residual is not None:
        loss_pde = F.mse_loss(pde_residual, torch.zeros_like(pde_residual))
    else:
        loss_pde = torch.zeros(1, device=device, requires_grad=True)


    return loss_pde
'''

'''
def pde_residual(u, grid_info, epsilon):
    """
    Calculates the residual of the 3D Allen-Cahn PDE using Fourier methods
    for spatial derivatives. This function is placed here for self-containment.

    The Allen-Cahn equation is: ∂u/∂t = ∇²u - (1/ε²)(u³ - u)
    The residual is defined as: R(u) = ∂u/∂t - ∇²u + (1/ε²)(u³ - u)

    Args:
        u (torch.Tensor): The field u in physical (denormalized) space.
                          Expected shape: (batch_size, Nx, Ny, Nz, T_total)
        grid_info (dict): Dictionary with 'Nx','Ny','Nz', 'Lx','Ly','Lz', 'dt_model'.
        epsilon (float): The interfacial energy parameter.

    Returns:
        torch.Tensor: The PDE residual at each point in space and time.
    """
    # 1. Unpack parameters
    batch_size, Nx, Ny, Nz, T_total = u.shape
    Lx, Ly, Lz = grid_info['Lx'], grid_info['Ly'], grid_info['Lz']
    dt = grid_info['dt_model']
    device = u.device

    # 2. Calculate the Laplacian term: ∇²u (in Fourier space)
    kx = torch.fft.fftfreq(Nx, d=Lx / Nx) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, d=Ly / Ny) * 2 * torch.pi
    kz = torch.fft.fftfreq(Nz, d=Lz / Nz) * 2 * torch.pi
    kxx, kyy, kzz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = (kxx ** 2 + kyy ** 2 + kzz ** 2).to(device)
    k_squared = k_squared.unsqueeze(0).unsqueeze(-1)  # Reshape for broadcasting

    u_hat = torch.fft.fftn(u, dim=[1, 2, 3])
    lap_u_hat = -k_squared * u_hat
    lap_u = torch.real(torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]))

    # 3. Calculate the time derivative term: ∂u/∂t (using finite differences)
    du_dt = torch.zeros_like(u)
    if T_total > 2:
        du_dt[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2 * dt)
    if T_total > 1:
        du_dt[..., 0] = (u[..., 1] - u[..., 0]) / dt  # Forward difference
        du_dt[..., -1] = (u[..., -1] - u[..., -2]) / dt  # Backward difference

    # 4. Calculate the nonlinear term: (1/ε²)(u³ - u)
    f_prime = (1.0 / (epsilon ** 2)) * (u ** 3 - u)

    # 5. Assemble the residual
    residual = du_dt - lap_u + f_prime

    # We only care about the residual for the predicted part of the trajectory
    return residual[..., 1:]  # Return residual for t > 0

'''

'''
# It's good practice to keep the pde_residual function here
def pde_residual(u, grid_info, epsilon):
    # ... (full function as provided in previous answers)
    device = u.device
    batch_size, Nx, Ny, Nz, T_total = u.shape
    Lx, Ly, Lz = grid_info['Lx'], grid_info['Ly'], grid_info['Lz']
    dt = grid_info['dt_model']
    kx = torch.fft.fftfreq(Nx, d=Lx / Nx) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, d=Ly / Ny) * 2 * torch.pi
    kz = torch.fft.fftfreq(Nz, d=Lz / Nz) * 2 * torch.pi
    kxx, kyy, kzz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = (kxx ** 2 + kyy ** 2 + kzz ** 2).to(device)
    k_squared = k_squared.unsqueeze(0).unsqueeze(-1)
    u_hat = torch.fft.fftn(u, dim=[1, 2, 3])
    lap_u_hat = -k_squared * u_hat
    lap_u = torch.real(torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]))
    du_dt = torch.zeros_like(u)
    if T_total > 2:
        du_dt[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2 * dt)
    if T_total > 1:
        du_dt[..., 0] = (u[..., 1] - u[..., 0]) / dt
        du_dt[..., -1] = (u[..., -1] - u[..., -2]) / dt
    f_prime = (1.0 / (epsilon ** 2)) * (u ** 3 - u)
    residual = du_dt - lap_u + f_prime
    return residual[..., 1:]
'''
# A helper function to avoid repeating the loss calculation logic

'''
## I trained model with this train hybrid function
def train_hybrid(model, myloss, adam_epochs, lbfgs_epochs, batch_size, train_loader, test_loader,
                 optimizer_adam, scheduler, normalized, normalizers, device,
                 pde_weight, grid_info, epsilon, problem, pde_loss_scaler):
    """
    A comprehensive training function with a two-phase optimization: Adam followed by L-BFGS.
    """
    # Initialize logs for the entire training process
    train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, test_pde_loss_scaled_log, test_pde_loss_unscaled_log = [], [], [], [], []
    train_data_log, train_pde_scaled_log, train_pde_unscaled_log, test_loss_hybrid_log = [], [], [], []

    # Ensure normalizers are on the correct device
    if normalized and normalizers:
        normalizers[0].to(device)
        normalizers[1].to(device)

    # =========================================================
    # =========== PHASE 1: ADAM OPTIMIZATION ==================
    # =========================================================
    print(f"\n--- Phase 1: Adam optimization for {adam_epochs} epochs ---")
    for epoch in range(adam_epochs):
        model.train()
        t1 = time.time()
        epoch_train_hybrid, epoch_train_data, epoch_train_pde, epoch_train_pde_unscl = 0, 0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Adam Epoch {epoch + 1}/{adam_epochs} [Training]"):
            x, y = x.to(device), y.to(device)
            optimizer_adam.zero_grad()

            total_loss, data_loss, scaled_pde_loss, pde_loss = _compute_hybrid_loss(
                model, x, y, myloss, normalizers, grid_info, pde_weight,
                pde_loss_scaler, epsilon, device
            )

            total_loss.backward()
            optimizer_adam.step()
            scheduler.step()

            epoch_train_hybrid += total_loss.item()
            epoch_train_data += data_loss.item()
            epoch_train_pde += scaled_pde_loss.item()
            epoch_train_pde_unscl += pde_loss.item()

        # Log training losses
        train_l2_hybrid_log.append(epoch_train_hybrid / len(train_loader))
        train_data_log.append(epoch_train_data / len(train_loader))
        train_pde_scaled_log.append(epoch_train_pde / len(train_loader))
        train_pde_unscaled_log.append(epoch_train_pde_unscl / len(train_loader))
        train_mse_hybrid_log.append(train_l2_hybrid_log[-1])

        # Validation loop
        model.eval()
        epoch_test_hybrid, epoch_test_data, epoch_test_pde, epoch_test_pde_unscl = 0, 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Adam Epoch {epoch + 1}/{adam_epochs} [Testing]"):
                x, y = x.to(device), y.to(device)
                total_loss, data_loss, scaled_pde_loss, pde_loss = _compute_hybrid_loss(
                    model, x, y, myloss, normalizers, grid_info, pde_weight,
                    pde_loss_scaler, epsilon, device
                )
                epoch_test_hybrid += total_loss.item()
                epoch_test_data += data_loss.item()
                epoch_test_pde += scaled_pde_loss.item()
                epoch_test_pde_unscl += pde_loss.item()

        test_loss_hybrid_log.append(epoch_test_hybrid / len(test_loader))
        test_data_log.append(epoch_test_data / len(test_loader))
        test_pde_loss_scaled_log.append(epoch_test_pde / len(test_loader))
        test_pde_loss_unscaled_log.append(epoch_test_pde_unscl / len(test_loader))

        t2 = time.time()

        print(f"Adam Epoch {epoch + 1}/{adam_epochs} | Time: {t2 - t1:.2f}s")
        print(
            f"  Train Hybrid Loss: {train_l2_hybrid_log[-1]:.4e} | Train Data Loss: {train_data_log[-1]:.4e} | Train PDE_scl. : {train_pde_scaled_log[-1]:.4e} | Train PDE_unscl. : {train_pde_unscaled_log[-1]:.4e}")
        print(
            f"  Test Hybrid Loss:  {test_loss_hybrid_log[-1]:.4e} | Test Data Loss:  {test_data_log[-1]:.4e} | Test PDE_scl:  {test_pde_loss_scaled_log[-1]:.4e} | Test PDE_unscl:  {test_pde_loss_unscaled_log[-1]:.4e}")
        print("-" * 80)


        #print(
        #    f"Adam Epoch {epoch + 1}/{adam_epochs} | Time: {t2 - t1:.2f}s | Train Loss: {train_l2_hybrid_log[-1]:.4e} | Test Loss: {test_loss_hybrid_log[-1]:.4e } | Test Pde_Scl. Loss: {test_pde_loss_scaled_log[-1]:.4e}")
        #print("-" * 80)

        # =========================================================
        # ========== PHASE 2: L-BFGS OPTIMIZATION (FIXED) =========
        # =========================================================
    if lbfgs_epochs > 0:
        print(f"\n--- Phase 2: L-BFGS optimization for {lbfgs_epochs} epochs ---")
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100,
                                            line_search_fn='strong_wolfe')

        # The closure function now uses gradient accumulation to save memory.
        def closure():
            # We must zero the gradients before accumulating them over the dataset.
            optimizer_lbfgs.zero_grad()

            # Accumulator for the scalar loss value.
            total_loss_value = 0.0

            # Loop through the entire training set to calculate accumulated gradients.
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Calculate loss for the current batch.
                total_loss, _, _, _ = _compute_hybrid_loss(
                    model, x, y, myloss, normalizers, grid_info,
                    pde_weight, pde_loss_scaler, epsilon, device
                )

                # --- THIS IS THE KEY FIX ---
                # To get the gradient of the mean loss over the dataset, we
                # backpropagate the loss of each batch, scaled by the number of batches.
                # This calculates gradients and immediately frees the computation graph,
                # preventing memory overload. The gradients are summed up in model.parameters().
                (total_loss / len(train_loader)).backward()

                # Accumulate the scalar value of the loss for returning.
                # .item() detaches it from the graph, preventing memory leaks.
                total_loss_value += total_loss.item()

            # Return the mean loss over the entire dataset.
            return total_loss_value / len(train_loader)

        # The L-BFGS training loop remains the same.
        for epoch in range(lbfgs_epochs):
            model.train()
            t1 = time.time()

            # The .step() call will now execute the memory-efficient closure.
            optimizer_lbfgs.step(closure)

            # After step, evaluate and log losses on full train and test sets for consistency
            model.eval()
            with torch.no_grad():
                # Training logs
                epoch_train_hybrid, epoch_train_data, epoch_train_pde = 0, 0, 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    total_loss, data_loss, scaled_pde_loss, pde_loss = _compute_hybrid_loss(model, x, y, myloss, normalizers,
                                                                                  grid_info, pde_weight,
                                                                                  pde_loss_scaler, epsilon, device)
                    epoch_train_hybrid += total_loss.item()
                    epoch_train_data += data_loss.item()
                    epoch_train_pde += scaled_pde_loss.item()
                train_l2_hybrid_log.append(epoch_train_hybrid / len(train_loader))
                train_data_log.append(epoch_train_data / len(train_loader))
                train_pde_scaled_log.append(epoch_train_pde / len(train_loader))
                train_mse_hybrid_log.append(train_l2_hybrid_log[-1])

                # Testing logs
                epoch_test_hybrid, epoch_test_data, epoch_test_pde = 0, 0, 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    total_loss, data_loss, scaled_pde_loss, pde_loss = _compute_hybrid_loss(model, x, y, myloss, normalizers,
                                                                                  grid_info, pde_weight,
                                                                                  pde_loss_scaler, epsilon, device)
                    epoch_test_hybrid += total_loss.item()
                    epoch_test_data += data_loss.item()
                    epoch_test_pde += scaled_pde_loss.item()
                test_loss_hybrid_log.append(epoch_test_hybrid / len(test_loader))
                test_data_log.append(epoch_test_data / len(test_loader))
                test_pde_loss_scaled_log.append(epoch_test_pde / len(test_loader))

            t2 = time.time()


            print(
                f"L-BFGS Epoch {epoch + 1}/{lbfgs_epochs} | Time: {t2 - t1:.2f}s | Train Loss: {train_l2_hybrid_log[-1]:.4e} | Test Loss: {test_loss_hybrid_log[-1]:.4e}")
            print("-" * 80)

        # Return the trained model and all the unified logs
        return (model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log,
                test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log)
'''
'''
# --- MAIN TRAINING FUNCTION ---
def train_hybrid(model, myloss, adam_epochs, lbfgs_epochs, batch_size, train_loader, test_loader,
                 optimizer_adam, scheduler, normalized, normalizers, device,
                 pde_weight, bc_weight, grid_info, epsilon, problem, pde_loss_scaler):
    """
    A comprehensive training function with a two-phase optimization: Adam followed by L-BFGS.
    Now includes boundary condition loss.
    """
    # Initialize logs for the entire training process
    train_mse_hybrid_log, train_l2_hybrid_log, test_data_log = [], [], []
    test_pde_loss_scaled_log, test_pde_loss_unscaled_log, test_bc_log = [], [], []
    train_data_log, train_pde_scaled_log, train_pde_unscaled_log, train_bc_log = [], [], [], []
    test_loss_hybrid_log = []

    # Ensure normalizers are on the correct device
    if normalized and normalizers:
        normalizers[0].to(device)
        normalizers[1].to(device)

    # =========================================================
    # =========== PHASE 1: ADAM OPTIMIZATION ==================
    # =========================================================
    print(f"\n--- Phase 1: Adam optimization for {adam_epochs} epochs ---")
    for epoch in range(adam_epochs):
        model.train()
        t1 = time.time()
        epoch_train_hybrid, epoch_train_data, epoch_train_pde_scl, epoch_train_pde_unscl, epoch_train_bc = 0, 0, 0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Adam Epoch {epoch + 1}/{adam_epochs} [Training]"):
            x, y = x.to(device), y.to(device)
            optimizer_adam.zero_grad()

            total_loss, data_loss, scaled_pde_loss, pde_loss, boundary_loss = _compute_hybrid_loss(
                model, x, y, myloss, normalizers, grid_info, pde_weight, bc_weight,
                pde_loss_scaler, epsilon, device
            )

            total_loss.backward()
            optimizer_adam.step()
            scheduler.step()

            epoch_train_hybrid += total_loss.item()
            epoch_train_data += data_loss.item()
            epoch_train_pde_scl += scaled_pde_loss.item()
            epoch_train_pde_unscl += pde_loss.item()
            epoch_train_bc += boundary_loss.item()

        # Log training losses
        train_l2_hybrid_log.append(epoch_train_hybrid / len(train_loader))
        train_data_log.append(epoch_train_data / len(train_loader))
        train_pde_scaled_log.append(epoch_train_pde_scl / len(train_loader))
        train_pde_unscaled_log.append(epoch_train_pde_unscl / len(train_loader))
        train_bc_log.append(epoch_train_bc / len(train_loader))
        train_mse_hybrid_log.append(train_l2_hybrid_log[-1])

        # Validation loop
        model.eval()
        epoch_test_hybrid, epoch_test_data, epoch_test_pde_scl, epoch_test_pde_unscl, epoch_test_bc = 0, 0, 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Adam Epoch {epoch + 1}/{adam_epochs} [Testing]"):
                x, y = x.to(device), y.to(device)
                total_loss, data_loss, scaled_pde_loss, pde_loss, boundary_loss = _compute_hybrid_loss(
                    model, x, y, myloss, normalizers, grid_info, pde_weight, bc_weight,
                    pde_loss_scaler, epsilon, device
                )
                epoch_test_hybrid += total_loss.item()
                epoch_test_data += data_loss.item()
                epoch_test_pde_scl += scaled_pde_loss.item()
                epoch_test_pde_unscl += pde_loss.item()
                epoch_test_bc += boundary_loss.item()

        test_loss_hybrid_log.append(epoch_test_hybrid / len(test_loader))
        test_data_log.append(epoch_test_data / len(test_loader))
        test_pde_loss_scaled_log.append(epoch_test_pde_scl / len(test_loader))
        test_pde_loss_unscaled_log.append(epoch_test_pde_unscl / len(test_loader))
        test_bc_log.append(epoch_test_bc / len(test_loader))

        t2 = time.time()
        print(f"Adam Epoch {epoch + 1}/{adam_epochs} | Time: {t2 - t1:.2f}s")
        print(
            f"  Train Hybrid: {train_l2_hybrid_log[-1]:.4e} | Data: {train_data_log[-1]:.4e} | PDE_scl: {train_pde_scaled_log[-1]:.4e} | BC: {train_bc_log[-1]:.4e}")
        print(
            f"  Test  Hybrid: {test_loss_hybrid_log[-1]:.4e} | Data: {test_data_log[-1]:.4e} | PDE_scl: {test_pde_loss_scaled_log[-1]:.4e} | BC: {test_bc_log[-1]:.4e}")
        print(f"  (Unscaled PDE Train: {train_pde_unscaled_log[-1]:.4e}, Test: {test_pde_loss_unscaled_log[-1]:.4e})")
        print("-" * 80)

    # =========================================================
    # ========== PHASE 2: L-BFGS OPTIMIZATION =================
    # =========================================================
    if lbfgs_epochs > 0:
        print(f"\n--- Phase 2: L-BFGS optimization for {lbfgs_epochs} epochs ---")
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100,
                                            line_search_fn='strong_wolfe')

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss_value = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                total_loss, _, _, _, _ = _compute_hybrid_loss(
                    model, x, y, myloss, normalizers, grid_info,
                    pde_weight, bc_weight, pde_loss_scaler, epsilon, device
                )
                (total_loss / len(train_loader)).backward()
                total_loss_value += total_loss.item()
            return total_loss_value / len(train_loader)

        for epoch in range(lbfgs_epochs):
            model.train()
            t1 = time.time()
            optimizer_lbfgs.step(closure)

            # After step, evaluate and log losses on full train and test sets
            model.eval()
            with torch.no_grad():
                # Training logs
                epoch_train_hybrid, epoch_train_data, epoch_train_pde_scl, epoch_train_bc = 0, 0, 0, 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    total_loss, data_loss, scaled_pde_loss, _, boundary_loss = _compute_hybrid_loss(
                        model, x, y, myloss, normalizers, grid_info, pde_weight, bc_weight,
                        pde_loss_scaler, epsilon, device
                    )
                    epoch_train_hybrid += total_loss.item()
                    epoch_train_data += data_loss.item()
                    epoch_train_pde_scl += scaled_pde_loss.item()
                    epoch_train_bc += boundary_loss.item()

                train_l2_hybrid_log.append(epoch_train_hybrid / len(train_loader))
                train_data_log.append(epoch_train_data / len(train_loader))
                train_pde_scaled_log.append(epoch_train_pde_scl / len(train_loader))
                train_bc_log.append(epoch_train_bc / len(train_loader))
                train_mse_hybrid_log.append(train_l2_hybrid_log[-1])

                # Testing logs
                epoch_test_hybrid, epoch_test_data, epoch_test_pde_scl, epoch_test_bc = 0, 0, 0, 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    total_loss, data_loss, scaled_pde_loss, _, boundary_loss = _compute_hybrid_loss(
                        model, x, y, myloss, normalizers, grid_info, pde_weight, bc_weight,
                        pde_loss_scaler, epsilon, device
                    )
                    epoch_test_hybrid += total_loss.item()
                    epoch_test_data += data_loss.item()
                    epoch_test_pde_scl += scaled_pde_loss.item()
                    epoch_test_bc += boundary_loss.item()

                test_loss_hybrid_log.append(epoch_test_hybrid / len(test_loader))
                test_data_log.append(epoch_test_data / len(test_loader))
                test_pde_loss_scaled_log.append(epoch_test_pde_scl / len(test_loader))
                test_bc_log.append(epoch_test_bc / len(test_loader))

            t2 = time.time()
            print(f"L-BFGS Epoch {epoch + 1}/{lbfgs_epochs} | Time: {t2 - t1:.2f}s")
            print(
                f"  Train Hybrid: {train_l2_hybrid_log[-1]:.4e} | Data: {train_data_log[-1]:.4e} | PDE_scl: {train_pde_scaled_log[-1]:.4e} | BC: {train_bc_log[-1]:.4e}")
            print(
                f"  Test  Hybrid: {test_loss_hybrid_log[-1]:.4e} | Data: {test_data_log[-1]:.4e} | PDE_scl: {test_pde_loss_scaled_log[-1]:.4e} | BC: {test_bc_log[-1]:.4e}")
            print("-" * 80)

    # Return the trained model and all the unified logs
    return (model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log,
            test_pde_loss_scaled_log, test_pde_loss_unscaled_log, test_bc_log,
            train_data_log, train_pde_scaled_log, train_pde_unscaled_log, train_bc_log,
            test_loss_hybrid_log)

'''

''''
def train_hybrid(model, myloss, epochs, batch_size, train_loader, test_loader,
                    optimizer, scheduler, normalized, normalizer, device, pde_weight, grid_info, epsilon, problem, pde_loss_scaler=1.0, can_compute_pde = True):
    train_mse_hybrid_log = []
    train_l2_hybrid_log = []

    test_mse_hybrid_log = []
    test_loss_hybrid_log = []


    train_data_log = []
    test_data_log = []

    train_pde_scaled_log = []
    train_pde_raw_log = []

    test_pde_loss_scaled_log = []
    test_pde_loss_raw_log = []


    if normalized:
        # a_normalizer = normalizer[0].to(device)
        y_normalizer = normalizer[1].to(device)
    else:
        # a_normalizer = None
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()

        train_mse_hybrid = 0
        train_l2_hybrid = 0

        train_data = 0.0
        train_pde_scaled = 0.0
        train_pde_raw = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            # print("x shape", x.shape)
            out = model(x)
            # print(f"Input shape: {x.shape}, y (target): {y.shape}, prediction (model output): {out.shape}")
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse_data = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
            loss_data = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

            # --- PDE Loss Calculation (on physical scale) ---
            if can_compute_pde:
                # loss_pde_raw = calculate_pde_residual_sh3d(pred_phys, grid_info, epsilon, device)
                #loss_pde_raw = calculate_pde_residual_sh3d(out, grid_info, epsilon, device)
                loss_pde_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                loss_pde_scaled = loss_pde_raw * pde_loss_scaler  # Apply scaling

            loss_hybrid = (1.0 - pde_weight) * loss_data + pde_weight * loss_pde_scaled
            mse_hybrid = (1.0 - pde_weight) * mse_data + pde_weight * loss_pde_scaled


            loss_hybrid.backward()
            optimizer.step()
            scheduler.step()

            # Hybrid
            train_mse_hybrid += mse_hybrid.item()
            train_l2_hybrid += loss_hybrid.item()
            ##
            train_data += loss_data.item()
            train_pde_scaled += loss_pde_scaled.item()
            train_pde_raw += loss_pde_raw.item()


        model.eval()
        test_data= 0.0 # data
        test_mse_data = 0.0
        test_pde_loss_scaled = 0.0
        test_pde_loss_raw = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                # test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
                test_mse_data += F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
                test_data += myloss(out.flatten(start_dim=1), y.flatten(start_dim=1)).item()

                # PDE Loss
                if can_compute_pde:
                    #loss_pde_test_raw = calculate_pde_residual_sh3d(out, grid_info, epsilon, device)
                    loss_pde_test_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)

                test_pde_loss_scaled += (loss_pde_test_raw * pde_loss_scaler).item()
                test_pde_loss_raw += loss_pde_test_raw.item()

            test_loss_hybrid = (1.0 - pde_weight) * test_data + pde_weight * test_pde_loss_scaled
            test_mse_hybrid = (1.0 - pde_weight) * test_mse_data + pde_weight * test_pde_loss_scaled

        # Train Hybrid
        train_mse_hybrid /= len(train_loader)
        train_l2_hybrid /= (batch_size * len(train_loader))

        # Test Hybrid
        test_mse_hybrid /= len(test_loader)
        test_loss_hybrid /= (batch_size * len(test_loader))

        # train data
        train_data /= (batch_size * len(train_loader))
        # test data
        test_data /= (batch_size * len(test_loader))
        # train pde
        train_pde_scaled /= (batch_size * len(train_loader))
        train_pde_raw /= (batch_size * len(train_loader))
        # test pde
        test_pde_loss_scaled /= (batch_size * len(test_loader))
        test_pde_loss_raw /= (batch_size * len(test_loader))




        # train Hybrid
        train_mse_hybrid_log.append(train_mse_hybrid)
        train_l2_hybrid_log.append(train_l2_hybrid)

        # Test Hybrid
        test_mse_hybrid_log.append(test_mse_hybrid)
        test_loss_hybrid_log.append(test_loss_hybrid)

        # train data
        train_data_log.append(train_data)
        # test data
        test_data_log.append(test_data)
        # train pde
        train_pde_scaled_log.append(train_pde_scaled)
        train_pde_raw_log.append(train_pde_raw)
        # test pde
        test_pde_loss_scaled_log.append(test_pde_loss_scaled)
        test_pde_loss_raw_log.append(test_pde_loss_raw)

        # Update the learning rate based on the test_l2 metric
        # scheduler.step(test_l2) ##

        t2 = default_timer()

        if ep == 0:
            # Update header to reflect spectral raw PDE loss
            print("No. Epoch | Time (s)   | Train MSE Hyb  | Train L2 Hyb  | test L2 Hyb | Test L2 data        | test_pde scl.   | test_pde_raw ")
            print("---------------------------------------------------------------------------------------------")
        # Update print statement
        print(f"{ep:<9}  {t2 - t1:<10.4f}   {train_mse_hybrid:<10.6e}     {train_l2_hybrid:<10.6e} {test_loss_hybrid:<10.6e}  {test_data:<24.6e} {test_pde_loss_scaled:<24.6e} {test_pde_loss_raw:<24.6e} ")

    return model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log
'''

''''
def train_hybrid(model, myloss, epochs, batch_size, train_loader, test_loader,
                    optimizer, scheduler, normalized, normalizer, device, init_pde_weight, grid_info,
                    epsilon, problem, pde_loss_scaler=1.0, can_compute_pde=True,
                    pde_weight_min=0.01, pde_weight_max=0.99, adaptivity_factor=0.1):
    train_mse_hybrid_log = []
    train_l2_hybrid_log = []
    test_mse_hybrid_log = []
    test_loss_hybrid_log = []
    train_data_log = []
    test_data_log = []
    train_pde_scaled_log = []
    train_pde_raw_log = []
    test_pde_loss_scaled_log = []
    test_pde_loss_raw_log = []

    # Initialize pde_weight as a float tensor on device (for gradient-free use)
    pde_weight = init_pde_weight

    if normalized:
        y_normalizer = normalizer[1].to(device)
    else:
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse_hybrid = 0
        train_l2_hybrid = 0
        train_data = 0.0
        train_pde_scaled = 0.0
        train_pde_raw = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)

            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse_data = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
            loss_data = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

            if can_compute_pde:
                loss_pde_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                loss_pde_scaled = loss_pde_raw * pde_loss_scaler
            else:
                loss_pde_raw = torch.tensor(0.0, device=device)
                loss_pde_scaled = torch.tensor(0.0, device=device)

            loss_hybrid = (1.0 - pde_weight) * loss_data + pde_weight * loss_pde_scaled
            mse_hybrid = (1.0 - pde_weight) * mse_data + pde_weight * loss_pde_scaled

            loss_hybrid.backward()
            optimizer.step()
            scheduler.step()

            train_mse_hybrid += mse_hybrid.item()
            train_l2_hybrid += loss_hybrid.item()
            train_data += loss_data.item()
            train_pde_scaled += loss_pde_scaled.item()
            train_pde_raw += loss_pde_raw.item()

        model.eval()
        test_data = 0.0
        test_mse_data = 0.0
        test_pde_loss_scaled = 0.0
        test_pde_loss_raw = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                test_mse_data += F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
                test_data += myloss(out.flatten(start_dim=1), y.flatten(start_dim=1)).item()
                if can_compute_pde:
                    loss_pde_test_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                else:
                    loss_pde_test_raw = torch.tensor(0.0, device=device)
                test_pde_loss_scaled += (loss_pde_test_raw * pde_loss_scaler).item()
                test_pde_loss_raw += loss_pde_test_raw.item()

            test_loss_hybrid = (1.0 - pde_weight) * test_data + pde_weight * test_pde_loss_scaled
            test_mse_hybrid = (1.0 - pde_weight) * test_mse_data + pde_weight * test_pde_loss_scaled

        # Average losses per batch
        train_mse_hybrid /= len(train_loader)
        train_l2_hybrid /= (batch_size * len(train_loader))
        test_mse_hybrid /= len(test_loader)
        test_loss_hybrid /= (batch_size * len(test_loader))
        train_data /= (batch_size * len(train_loader))
        test_data /= (batch_size * len(test_loader))
        train_pde_scaled /= (batch_size * len(train_loader))
        train_pde_raw /= (batch_size * len(train_loader))
        test_pde_loss_scaled /= (batch_size * len(test_loader))
        test_pde_loss_raw /= (batch_size * len(test_loader))

        train_mse_hybrid_log.append(train_mse_hybrid)
        train_l2_hybrid_log.append(train_l2_hybrid)
        test_mse_hybrid_log.append(test_mse_hybrid)
        test_loss_hybrid_log.append(test_loss_hybrid)
        train_data_log.append(train_data)
        test_data_log.append(test_data)
        train_pde_scaled_log.append(train_pde_scaled)
        train_pde_raw_log.append(train_pde_raw)
        test_pde_loss_scaled_log.append(test_pde_loss_scaled)
        test_pde_loss_raw_log.append(test_pde_loss_raw)

        # Adaptive PDE weight update: simple proportional adjustment by relative losses ratio
        if train_pde_scaled > 0 and train_data > 0:
            ratio = train_data / (train_pde_scaled + 1e-8)  # Prevent div 0
            # Update rule: increase PDE weight if data loss much larger, decrease if PDE loss dominates
            new_pde_weight = pde_weight + adaptivity_factor * (ratio - 1)
            # Clamp weight to specified min/max bounds
            pde_weight = max(pde_weight_min, min(pde_weight_max, new_pde_weight))

        t2 = default_timer()
        if ep == 0:
            print("No. Epoch | Time (s)   | Train MSE Hyb  | Train L2 Hyb  | test L2 Hyb | Test L2 data        | test_pde scl.   | test_pde_raw ")
            print("---------------------------------------------------------------------------------------------")
        print(f"{ep:<9}  {t2 - t1:<10.4f}   {train_mse_hybrid:<10.6e}     {train_l2_hybrid:<10.6e} {test_loss_hybrid:<10.6e}  {test_data:<24.6e} {test_pde_loss_scaled:<24.6e} {test_pde_loss_raw:<24.6e} ")
        print(f"Adaptive PDE weight after epoch {ep}: {pde_weight:.4f}")

    return model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log
'''

'''
# pde weight = 0.3
def train_hybrid(model, myloss, epochs, batch_size, train_loader, test_loader,
                 optimizer, scheduler, normalized, normalizer, device, init_pde_weight,
                 grid_info, epsilon, problem, pde_loss_scaler=1.0, can_compute_pde=True,
                 pde_weight_min=0.1, pde_weight_max=0.6, adaptivity_factor=0.05, warmup_epochs=5):
    train_mse_hybrid_log = []
    train_l2_hybrid_log = []
    test_mse_hybrid_log = []
    test_loss_hybrid_log = []
    train_data_log = []
    test_data_log = []
    train_pde_scaled_log = []
    train_pde_raw_log = []
    test_pde_loss_scaled_log = []
    test_pde_loss_raw_log = []

    pde_weight = init_pde_weight
    # Use exponential moving average for smooth updates to losses
    ema_alpha = 0.7
    ema_train_data = None
    ema_train_pde = None

    if normalized:
        y_normalizer = normalizer[1].to(device)
    else:
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse_hybrid = 0
        train_l2_hybrid = 0
        train_data = 0.0
        train_pde_scaled = 0.0
        train_pde_raw = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)

            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse_data = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
            loss_data = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

            if can_compute_pde:
                loss_pde_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                loss_pde_scaled = loss_pde_raw * pde_loss_scaler
            else:
                loss_pde_raw = torch.tensor(0.0, device=device)
                loss_pde_scaled = torch.tensor(0.0, device=device)

            # During warmup, gradually ramp up PDE loss weight linearly
            if ep < warmup_epochs:
                current_pde_weight = pde_weight * (ep + 1) / warmup_epochs
            else:
                current_pde_weight = pde_weight

            loss_hybrid = (1.0 - current_pde_weight) * loss_data + current_pde_weight * loss_pde_scaled
            mse_hybrid = (1.0 - current_pde_weight) * mse_data + current_pde_weight * loss_pde_scaled

            loss_hybrid.backward()
            optimizer.step()
            scheduler.step()

            train_mse_hybrid += mse_hybrid.item()
            train_l2_hybrid += loss_hybrid.item()
            train_data += loss_data.item()
            train_pde_scaled += loss_pde_scaled.item()
            train_pde_raw += loss_pde_raw.item()

        # Compute averages per batch
        train_data_avg = train_data / (batch_size * len(train_loader))
        train_pde_scaled_avg = train_pde_scaled / (batch_size * len(train_loader))

        # Update EMA values
        if ema_train_data is None:
            ema_train_data = train_data_avg
            ema_train_pde = train_pde_scaled_avg
        else:
            ema_train_data = ema_alpha * train_data_avg + (1 - ema_alpha) * ema_train_data
            ema_train_pde = ema_alpha * train_pde_scaled_avg + (1 - ema_alpha) * ema_train_pde

        model.eval()
        test_data = 0.0
        test_mse_data = 0.0
        test_pde_loss_scaled = 0.0
        test_pde_loss_raw = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                test_mse_data += F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
                test_data += myloss(out.flatten(start_dim=1), y.flatten(start_dim=1)).item()
                if can_compute_pde:
                    loss_pde_test_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                else:
                    loss_pde_test_raw = torch.tensor(0.0, device=device)
                test_pde_loss_scaled += (loss_pde_test_raw * pde_loss_scaler).item()
                test_pde_loss_raw += loss_pde_test_raw.item()

            test_loss_hybrid = (1.0 - current_pde_weight) * (test_data / (batch_size * len(test_loader))) + \
                               current_pde_weight * (test_pde_loss_scaled / (batch_size * len(test_loader)))
            test_mse_hybrid = (1.0 - current_pde_weight) * (test_mse_data / len(test_loader)) + \
                              current_pde_weight * (test_pde_loss_scaled / (batch_size * len(test_loader)))

        # Average train losses
        train_mse_hybrid /= len(train_loader)
        train_l2_hybrid /= (batch_size * len(train_loader))

        # Logging for plotting and analysis
        train_mse_hybrid_log.append(train_mse_hybrid)
        train_l2_hybrid_log.append(train_l2_hybrid)
        test_mse_hybrid_log.append(test_mse_hybrid)
        test_loss_hybrid_log.append(test_loss_hybrid)
        train_data_log.append(train_data_avg)
        test_data_log.append(test_data / (batch_size * len(test_loader)))
        train_pde_scaled_log.append(train_pde_scaled_avg)
        train_pde_raw_log.append(train_pde_raw / (batch_size * len(train_loader)))
        test_pde_loss_scaled_log.append(test_pde_loss_scaled / (batch_size * len(test_loader)))
        test_pde_loss_raw_log.append(test_pde_loss_raw / (batch_size * len(test_loader)))

        # Adaptive PDE weight update (starting after warmup)
        if ep >= warmup_epochs and ema_train_data > 0 and ema_train_pde > 0:
            ratio = ema_train_data / (ema_train_pde + 1e-8)
            # Smooth update: difference scaled by adaptivity factor and weighted moving average
            new_pde_weight = pde_weight + adaptivity_factor * (ratio - 1)
            pde_weight = max(pde_weight_min, min(pde_weight_max, new_pde_weight))

        t2 = default_timer()

        if ep == 0:
            print("No. Epoch | Time (s)   | Train MSE Hyb  | Train L2 Hyb  | test L2 Hyb | Test L2 data        | test_pde scl.   | test_pde_raw | PDE Weight")
            print("---------------------------------------------------------------------------------------------------------------")
        print(f"{ep:<9}  {t2 - t1:<10.4f}   {train_mse_hybrid:<10.6e}     {train_l2_hybrid:<10.6e} {test_loss_hybrid:<10.6e}  {test_data_log[-1]:<24.6e} {test_pde_loss_scaled_log[-1]:<24.6e} {test_pde_loss_raw_log[-1]:<12.6e} {pde_weight:.4f}")

    return model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, \
           test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log
'''

'''
# this seem good!
def train_hybrid(model, myloss, epochs, batch_size, train_loader, test_loader,
                 optimizer, scheduler, normalized, normalizer, device, init_pde_weight,
                 grid_info, epsilon, problem, pde_loss_scaler=1.0, can_compute_pde=True,
                 pde_weight_min=0.1, pde_weight_max=0.6, adaptivity_factor=0.05, warmup_epochs=5,
                 target_pde_lo=1e-2, target_pde_hi=1e-1):
    import math

    train_mse_hybrid_log = []
    train_l2_hybrid_log = []
    test_mse_hybrid_log = []
    test_loss_hybrid_log = []
    train_data_log = []
    test_data_log = []
    train_pde_scaled_log = []
    train_pde_raw_log = []
    test_pde_loss_scaled_log = []
    test_pde_loss_raw_log = []

    pde_weight = init_pde_weight
    data_weight = 1.0 - pde_weight
    ema_alpha = 0.7
    ema_train_data = None
    ema_train_pde = None
    ema_test_pde = None

    if normalized:
        y_normalizer = normalizer[1].to(device)
    else:
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        from timeit import default_timer
        t1 = default_timer()
        train_mse_hybrid = 0
        train_l2_hybrid = 0
        train_data = 0.0
        train_pde_scaled = 0.0
        train_pde_raw = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)

            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse_data = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
            loss_data = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

            if can_compute_pde:
                loss_pde_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                loss_pde_scaled = loss_pde_raw * pde_loss_scaler
            else:
                loss_pde_raw = torch.tensor(0.0, device=device)
                loss_pde_scaled = torch.tensor(0.0, device=device)

            # During warmup, gradually ramp up PDE loss weight linearly
            if ep < warmup_epochs:
                current_pde_weight = pde_weight * (ep + 1) / warmup_epochs
                current_data_weight = 1.0 - current_pde_weight
            else:
                current_pde_weight = pde_weight
                current_data_weight = 1.0 - current_pde_weight

            loss_hybrid = current_data_weight * loss_data + current_pde_weight * loss_pde_scaled
            mse_hybrid = current_data_weight * mse_data + current_pde_weight * loss_pde_scaled

            loss_hybrid.backward()
            optimizer.step()
            scheduler.step()

            train_mse_hybrid += mse_hybrid.item()
            train_l2_hybrid += loss_hybrid.item()
            train_data += loss_data.item()
            train_pde_scaled += loss_pde_scaled.item()
            train_pde_raw += loss_pde_raw.item()

        train_data_avg = train_data / (batch_size * len(train_loader))
        train_pde_scaled_avg = train_pde_scaled / (batch_size * len(train_loader))

        if ema_train_data is None:
            ema_train_data = train_data_avg
            ema_train_pde = train_pde_scaled_avg
        else:
            ema_train_data = ema_alpha * train_data_avg + (1 - ema_alpha) * ema_train_data
            ema_train_pde = ema_alpha * train_pde_scaled_avg + (1 - ema_alpha) * ema_train_pde

        model.eval()
        test_data = 0.0
        test_mse_data = 0.0
        test_pde_loss_scaled = 0.0
        test_pde_loss_raw = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                test_mse_data += F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
                test_data += myloss(out.flatten(start_dim=1), y.flatten(start_dim=1)).item()
                if can_compute_pde:
                    loss_pde_test_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                else:
                    loss_pde_test_raw = torch.tensor(0.0, device=device)
                test_pde_loss_scaled += (loss_pde_test_raw * pde_loss_scaler).item()
                test_pde_loss_raw += loss_pde_test_raw.item()

            test_loss_hybrid = current_data_weight * (test_data / (batch_size * len(test_loader))) + \
                               current_pde_weight * (test_pde_loss_scaled / (batch_size * len(test_loader)))
            test_mse_hybrid = current_data_weight * (test_mse_data / len(test_loader)) + \
                              current_pde_weight * (test_pde_loss_scaled / (batch_size * len(test_loader)))

        train_mse_hybrid /= len(train_loader)
        train_l2_hybrid /= (batch_size * len(train_loader))

        train_mse_hybrid_log.append(train_mse_hybrid)
        train_l2_hybrid_log.append(train_l2_hybrid)
        test_mse_hybrid_log.append(test_mse_hybrid)
        test_loss_hybrid_log.append(test_loss_hybrid)
        train_data_log.append(train_data_avg)
        test_data_log.append(test_data / (batch_size * len(test_loader)))
        train_pde_scaled_log.append(train_pde_scaled_avg)
        train_pde_raw_log.append(train_pde_raw / (batch_size * len(train_loader)))
        test_pde_loss_scaled_log.append(test_pde_loss_scaled / (batch_size * len(test_loader)))
        test_pde_loss_raw_log.append(test_pde_loss_raw / (batch_size * len(test_loader)))

        # Get current average test PDE loss
        current_test_pde = test_pde_loss_scaled / (batch_size * len(test_loader))
        if ep >= warmup_epochs:
            if ema_test_pde is None:
                ema_test_pde = current_test_pde
            else:
                ema_test_pde = ema_alpha * current_test_pde + (1 - ema_alpha) * ema_test_pde

            # Adjust PDE weight to keep PDE loss roughly within target range
            if ema_test_pde < target_pde_lo:
                # PDE loss is low, so decrease PDE weight and increase Data weight
                pde_weight -= adaptivity_factor
            elif ema_test_pde > target_pde_hi:
                # PDE loss is high, increase PDE weight and decrease Data weight
                pde_weight += adaptivity_factor

            # Clamp PDE weight and update Data weight accordingly
            pde_weight = max(pde_weight_min, min(pde_weight_max, pde_weight))
            data_weight = 1.0 - pde_weight

        t2 = default_timer()

        if ep == 0:
            print("No. Epoch | Time (s)   | Train MSE Hyb  | Train L2 Hyb  | test L2 Hyb | Test L2 data        | test_pde scl.   | test_pde_raw | PDE Weight | Data Weight")
            print("-----------------------------------------------------------------------------------------------------------------")

        print(f"{ep:<9}  {t2 - t1:<10.4f}   {train_mse_hybrid:<10.6e}     {train_l2_hybrid:<10.6e} {test_loss_hybrid:<10.6e}  "
              f"{test_data_log[-1]:<24.6e} {test_pde_loss_scaled_log[-1]:<24.6e} {test_pde_loss_raw_log[-1]:<12.6e} "
              f"{pde_weight:.4f}     {data_weight:.4f}")

    return model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, \
           test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log

'''

''''
def train_hybrid(
    model, myloss, epochs, batch_size, train_loader, test_loader,
    optimizer, scheduler, normalized, normalizer, device, pde_weight,
    grid_info, epsilon, problem, init_pde_loss_scaler=1e-2,
    can_compute_pde=True, scale_update_interval=3, scale_lag=2, scale_update_factor=5.0,
    min_scaler=1e-5, max_scaler=1e2, scaler_balance_tol=0.33
):
    import torch.nn.functional as F
    from timeit import default_timer

    # Logs
    train_mse_hybrid_log, train_l2_hybrid_log = [], []
    test_mse_hybrid_log, test_loss_hybrid_log = [], []
    train_data_log, test_data_log = [], []
    train_pde_scaled_log, train_pde_raw_log = [], []
    test_pde_loss_scaled_log, test_pde_loss_raw_log = [], []

    if normalized:
        y_normalizer = normalizer[1].to(device)
    else:
        y_normalizer = None

    pde_loss_scaler = init_pde_loss_scaler
    old_test_data = None
    old_test_pde = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        # Training
        train_mse_hybrid = train_l2_hybrid = 0.0
        train_data = train_pde_scaled = train_pde_raw = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            mse_data = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
            loss_data = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

            if can_compute_pde:
                loss_pde_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                loss_pde_scaled = loss_pde_raw * pde_loss_scaler
            else:
                loss_pde_raw = torch.tensor(0.0, device=device)
                loss_pde_scaled = torch.tensor(0.0, device=device)

            loss_hybrid = (1.0 - pde_weight) * loss_data + pde_weight * loss_pde_scaled
            mse_hybrid = (1.0 - pde_weight) * mse_data + pde_weight * loss_pde_scaled

            loss_hybrid.backward()
            optimizer.step()
            scheduler.step()
            # accumulate
            train_mse_hybrid += mse_hybrid.item()
            train_l2_hybrid += loss_hybrid.item()
            train_data += loss_data.item()
            train_pde_scaled += loss_pde_scaled.item()
            train_pde_raw += loss_pde_raw.item()

        train_mse_hybrid /= len(train_loader)
        train_l2_hybrid /= (batch_size * len(train_loader))
        train_data_avg = train_data / (batch_size * len(train_loader))
        train_pde_scaled_avg = train_pde_scaled / (batch_size * len(train_loader))
        train_pde_raw_avg = train_pde_raw / (batch_size * len(train_loader))

        # Testing
        model.eval()
        test_data = 0.0
        test_mse_data = 0.0
        test_pde_loss_scaled = 0.0
        test_pde_loss_raw = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                mse_data_t = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
                loss_data_t = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))
                if can_compute_pde:
                    loss_pde_raw_t = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                    loss_pde_scaled_t = loss_pde_raw_t * pde_loss_scaler
                else:
                    loss_pde_raw_t = torch.tensor(0.0, device=device)
                    loss_pde_scaled_t = torch.tensor(0.0, device=device)
                test_mse_data += mse_data_t
                test_data += loss_data_t.item()
                test_pde_loss_scaled += loss_pde_scaled_t.item()
                test_pde_loss_raw += loss_pde_raw_t.item()

        test_data_avg = test_data / (batch_size * len(test_loader))
        test_pde_scaled_avg = test_pde_loss_scaled / (batch_size * len(test_loader))
        test_pde_raw_avg = test_pde_loss_raw / (batch_size * len(test_loader))
        test_mse_data_avg = test_mse_data / len(test_loader)

        test_loss_hybrid = (1.0 - pde_weight) * test_data_avg + pde_weight * test_pde_scaled_avg
        test_mse_hybrid = (1.0 - pde_weight) * test_mse_data_avg + pde_weight * test_pde_scaled_avg

        # Logging for monitoring and plotting
        train_mse_hybrid_log.append(train_mse_hybrid)
        train_l2_hybrid_log.append(train_l2_hybrid)
        test_mse_hybrid_log.append(test_mse_hybrid)
        test_loss_hybrid_log.append(test_loss_hybrid)
        train_data_log.append(train_data_avg)
        test_data_log.append(test_data_avg)
        train_pde_scaled_log.append(train_pde_scaled_avg)
        train_pde_raw_log.append(train_pde_raw_avg)
        test_pde_loss_scaled_log.append(test_pde_scaled_avg)
        test_pde_loss_raw_log.append(test_pde_raw_avg)

        # === Dynamic Loss Scaling (to keep both losses similar in scale) ===
        # Every few epochs (with optional lag for stability)
        if (ep >= (scale_lag+1)) and (ep % scale_update_interval == 0):
            recent_pde = np.mean(test_pde_loss_scaled_log[max(0,ep-scale_lag):ep+1])
            recent_data = np.mean(test_data_log[max(0,ep-scale_lag):ep+1])
            # If PDE loss is much smaller than data loss, increase scaler
            if recent_pde < recent_data * scaler_balance_tol:
                pde_loss_scaler = min(max_scaler, pde_loss_scaler * scale_update_factor)
                print(f">>>> physics too small, increasing scaler to {pde_loss_scaler:.2e}")
            # If data loss is much smaller than PDE loss, decrease scaler
            elif recent_data < recent_pde * scaler_balance_tol:
                pde_loss_scaler = max(min_scaler, pde_loss_scaler / scale_update_factor)
                print(f">>>> data too small, decreasing scaler to {pde_loss_scaler:.2e}")

        t2 = default_timer()
        if ep == 0:
            print("No. Epoch | Time (s)   | Train MSE Hyb| Train L2 Hyb | test L2 Hyb | Test L2 data        | test_pde scl.   | test_pde_raw | pde_loss_scaler")
            print("----------------------------------------------------------------------------------------------------------------")
        print(f"{ep:<9}  {t2 - t1:<10.4f} {train_mse_hybrid:<13.6e} {train_l2_hybrid:<12.6e} {test_loss_hybrid:<12.6e} "
              f"{test_data_avg:<20.6e} {test_pde_scaled_avg:<20.6e} {test_pde_raw_avg:<13.6e} {pde_loss_scaler:.5e}")

    return model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, \
           test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log

'''

def train_fno4d(model, myloss, epochs, batch_size, train_loader, test_loader,
                optimizer, scheduler, normalized, normalizer, device):
    """Training function for FNO4d model."""
    ntrain = len(train_loader) * train_loader.batch_size
    ntest = len(test_loader) * test_loader.batch_size

    train_mse_log = []
    train_l2_log = []
    test_l2_log = []
    test_mse_log = []

    if normalized:
        y_normalizer = normalizer[1].to(device)
    else:
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')
            loss = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_mse += mse.item()
            #train_l2 += loss.item()
            train_l2 += loss.item() / batch_size  # Normalize by batch size

        model.eval()
        test_l2 = 0.0
        test_mse = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_mse += F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean').item()
                test_l2 += myloss(out.flatten(start_dim=1), y.flatten(start_dim=1)).item()/ batch_size
                #test_l2 = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))
                #test_l2 += test_l2.item() / batch_size


        train_mse /= len(train_loader)
        train_l2 /= len(train_loader)
        test_mse /= len(test_loader)
        test_l2 /= len(test_loader)

        train_mse_log.append(train_mse)
        train_l2_log.append(train_l2)
        test_l2_log.append(test_l2)
        test_mse_log.append(test_mse)

        t2 = default_timer()

        if ep == 0:
            print("No. Epoch   Time (s)       Train MSE      Train L2            Test L2")

        print(f"{ep:<10} {t2 - t1:<13.6f} {train_mse:<13.10f} {train_l2:<13.10f}  {test_l2:<13.10f}")

    return model, train_mse_log, train_l2_log, test_l2_log

###############################
###############################

# ------------------------- FNO4d hybrid trainer (discrete physics) -------------------------
import torch
import torch.nn.functional as F
import numpy as np
from timeit import default_timer


def _spectral_k2(Nx, Ny, Nz, Lx, Ly, Lz, device):
    hx = Lx / Nx; hy = Ly / Ny; hz = Lz / Nz
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=hx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=hy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=hz)
    kxx, kyy, kzz = np.meshgrid(kx**2, ky**2, kz**2, indexing="ij")
    k2 = (kxx + kyy + kzz).astype(np.float32)
    return torch.as_tensor(k2, device=device, dtype=torch.complex64)


def _laplacian_spectral(u, k2_complex):
    # u: (B, s, s, s) real -> Δu (B, s, s, s) real
    u_fft = torch.fft.fftn(u, dim=(1, 2, 3))
    lap_fft = -k2_complex * u_fft
    return torch.fft.ifftn(lap_fft, dim=(1, 2, 3)).real


def _periodic_bc_loss(u):
    # u: (B, s, s, s)
    loss_x = F.mse_loss(u[:, 0, :, :],  u[:, -1, :, :])
    loss_y = F.mse_loss(u[:, :, 0, :],  u[:, :, -1, :])
    loss_z = F.mse_loss(u[:, :, :, 0],  u[:, :, :, -1])
    bc_val = (loss_x + loss_y + loss_z) / 3.0

    dx0 = u[:, 1, :, :]  - u[:, 0, :, :]
    dxN = u[:, -1, :, :] - u[:, -2, :, :]
    dy0 = u[:, :, 1, :]  - u[:, :, 0, :]
    dyN = u[:, :, -1, :] - u[:, :, -2, :]
    dz0 = u[:, :, :, 1]  - u[:, :, :, 0]
    dzN = u[:, :, :, -1] - u[:, :, :, -2]
    bc_grad = (F.mse_loss(dx0, dxN) + F.mse_loss(dy0, dyN) + F.mse_loss(dz0, dzN)) / 3.0
    return bc_val, bc_grad


def train_hybrid_fno4d(
    model, myloss, epochs, train_loader, test_loader,
    optimizer, scheduler, normalized, normalizers, device,
    pde_weight=0.25,
    grid_info=None,
    pde_loss_scaler=1.0,
    lambda_bc_val=1e-3,
    lambda_bc_grad=0.0,
):
    """
    One-step FNO4d hybrid trainer (Allen–Cahn dataset generated by semi-implicit scheme)

    Inputs/targets from your pipeline:
      x: (B, s, s, s, T_in)
      y: (B, s, s, s, T_out)   # we supervise only the first future step y[..., 0:1]
      out = model(x): (B, s, s, s, 1)

    Discrete residual (matching your generator):
      r = u_next - dt*Δ u_next - u_prev + (dt/Cahn) * (u_prev^3 - u_prev)

    grid_info expects:
      {'Nx','Ny','Nz','Lx','Ly','Lz','dt_model','EPSILON_PARAM' or 'CAHN'}
      NOTE: CAHN = epsilon^2 in your MATLAB (i.e., 0.01 if epsilon=0.1).
    """
    assert grid_info is not None, "grid_info dict is required"

    s  = int(grid_info["Nx"])
    Lx = float(grid_info["Lx"]); Ly = float(grid_info.get("Ly", Lx)); Lz = float(grid_info.get("Lz", Lx))
    dt = float(grid_info["dt_model"])
    # Prefer explicit CAHN if provided; otherwise infer from EPSILON_PARAM
    if "CAHN" in grid_info:
        CAHN = float(grid_info["CAHN"])
    else:
        eps = float(grid_info.get("EPSILON_PARAM", 0.1))  # epsilon
        CAHN = eps**2                                     # Cahn = epsilon^2

    k2_complex = _spectral_k2(s, s, s, Lx, Ly, Lz, device)

    a_normalizer = normalizers[0].to(device) if (normalized and normalizers and normalizers[0] is not None) else None
    y_normalizer = normalizers[1].to(device) if (normalized and normalizers and normalizers[1] is not None) else None

    train_total_log, train_data_log, train_pde_scaled_log, test_l2_log = [], [], [], []

    print("\nNo. Epoch   Time (s)       Train(data)    Train(PDEs)     Train(total)   Test(data)")
    print("--------------------------------------------------------------------------")
    print("  ep        dt(s)        MSE(data)      Phys(scaled)     Total(hybrid)    Val(data)")

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        sum_data = 0.0
        sum_phys = 0.0
        nb = 0

        for x, y in train_loader:
            nb += 1
            x = x.to(device)                           # (B,s,s,s,T_in)
            y = y.to(device)                           # (B,s,s,s,T_out)
            y_next = y[..., 0:1]                       # (B,s,s,s,1) one-step supervision

            optimizer.zero_grad()

            out = model(x)                             # (B,s,s,s,1)

            # ---- decode to physical scale ----
            if y_normalizer is not None:
                out_phys = y_normalizer.decode(out)
                y_phys   = y_normalizer.decode(y_next)
            else:
                out_phys = out
                y_phys   = y_next

            data_loss = F.mse_loss(
                out_phys.flatten(start_dim=1),
                y_phys.flatten(start_dim=1),
                reduction='mean'
            )

            # ---- physics (discrete residual, teacher forcing on u_prev) ----
            u_next = out_phys[..., 0]                  # (B,s,s,s)
            if a_normalizer is not None:
                x_phys = a_normalizer.decode(x)        # (B,s,s,s,T_in)
                u_prev = x_phys[..., -1]               # last known frame
            else:
                u_prev = x[..., -1]

            lap_u_next = _laplacian_spectral(u_next, k2_complex)

            # r = u_next - dt*Δ u_next - u_prev + (dt/CAHN)*(u_prev^3 - u_prev)
            r = u_next - dt * lap_u_next - u_prev + (dt / CAHN) * (u_prev**3 - u_prev)
            pde_loss = torch.mean(r**2)

            # periodic BC (tiny weight – spectral update already enforces periodicity)
            bc_val, bc_grad = _periodic_bc_loss(u_next)
            physics_loss = pde_loss_scaler * pde_loss + lambda_bc_val * bc_val + lambda_bc_grad * bc_grad

            total = (1.0 - pde_weight) * data_loss + pde_weight * physics_loss

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            sum_data += data_loss.item()
            sum_phys += physics_loss.item()

        # epoch means
        mean_data = sum_data / max(1, nb)
        mean_phys = sum_phys / max(1, nb)
        mean_total = (1.0 - pde_weight) * mean_data + pde_weight * mean_phys

        # ---- validation (data only, one-step) ----
        model.eval()
        with torch.no_grad():
            val_sum = 0.0
            vn = 0
            for xv, yv in test_loader:
                vn += 1
                xv = xv.to(device)
                yv = yv.to(device)
                yv_next = yv[..., 0:1]
                ov = model(xv)
                if y_normalizer is not None:
                    ov      = y_normalizer.decode(ov)
                    yv_next = y_normalizer.decode(yv_next)
                val_sum += F.mse_loss(ov.flatten(start_dim=1), yv_next.flatten(start_dim=1), reduction='mean').item()
            val_mean = val_sum / max(1, vn)

        t2 = default_timer()
        print(f"{ep:4d}    {t2 - t1:10.4f}    {mean_data:12.6e}  {mean_phys:12.6e}  "
              f"{mean_total:12.6e}  {val_mean:12.6e}")

        train_data_log.append(mean_data)
        train_pde_scaled_log.append(mean_phys)
        train_total_log.append(mean_total)
        test_l2_log.append(val_mean)

    return model, train_total_log, train_data_log, train_pde_scaled_log, test_l2_log
