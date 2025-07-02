import torch
import torch.nn.functional as F
from timeit import default_timer
from tqdm import tqdm

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



########################
########################

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
        rhs_ac3d = Cahn_ac * lap_u - f_prime_u
        pde_residual = du_dt - rhs_ac3d

    elif problem == 'CH3D':
        Cahn_ch = epsilon #  pde_params.get('Cahn_ch')
        if Cahn_ch is None: raise ValueError("Parameter 'Cahn_ch' not provided for CH3D.")
        lap_u_hat = -k2_m.unsqueeze(0).unsqueeze(-1) * u_hat
        lap_u = torch.fft.ifftn(lap_u_hat, dim=[1, 2, 3]).real
        #mu_terms = (u_phys**3 - 3 * u_phys) - Cahn_ch * lap_u
        mu_terms = (u_phys ** 3 -  u_phys) - Cahn_ch * lap_u
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
        rhs_mbe3d = -lap_u + epsilon_mbe * biharm_u - div_term
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
        rhs_pfc3d = -(1 - epsilon_pfc) * lap_u - 2 * biharm_u - triharm_u - lap_u_cubed
        pde_residual = du_dt - rhs_pfc3d

    else:
        raise ValueError(f"Unknown problem_name: {problem}. PDE residual not defined.")

    if pde_residual is not None:
        loss_pde = F.mse_loss(pde_residual, torch.zeros_like(pde_residual))
    else:
        loss_pde = torch.zeros(1, device=device, requires_grad=True)

    return loss_pde


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


def compute_initial_loss_scaler(model, loader, myloss, normalized, normalizer, device, grid_info, epsilon, problem):
    """
    Computes the scaling factor to balance data and PDE losses.
    """
    model.eval()

    # Get one batch from the loader
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        out = model(x)
        if normalized:
            y_normalizer = normalizer[1].to(device)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

        # Calculate initial data loss
        initial_loss_data = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))

        # Calculate initial raw PDE loss
        initial_loss_pde_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)

        # Handle case where PDE loss is zero to avoid division by zero
        if initial_loss_pde_raw.item() < 1e-12:
            scaler = 1.0
            print("Warning: Initial PDE loss is near zero. Setting scaler to 1.0.")
        else:
            # The scaler is the ratio of the two losses
            scaler = initial_loss_data.item() / initial_loss_pde_raw.item()
            print(f"Computed initial loss scaler: {scaler:.4f}")
            print(f"  - Initial Data Loss: {initial_loss_data.item():.6f}")
            print(f"  - Initial PDE Loss (raw): {initial_loss_pde_raw.item():.6f}")

    model.train()  # Set model back to training mode
    return scaler


''''
def train_hybrid(model, myloss, epochs, batch_size, train_loader, test_loader,
                 optimizer, scheduler, normalized, normalizer, device, pde_weight,
                 grid_info, epsilon, problem, pde_loss_scaler='auto', can_compute_pde=True):
    # --- Logging Lists ---
    train_mse_hybrid_log, train_l2_hybrid_log = [], []
    test_mse_hybrid_log, test_loss_hybrid_log = [], []
    train_data_log, test_data_log = [], []
    train_pde_scaled_log, train_pde_raw_log = [], []
    test_pde_loss_scaled_log, test_pde_loss_raw_log = [], []

    if normalized:
        y_normalizer = normalizer[1].to(device)
    else:
        y_normalizer = None

    # ====================================================================================
    # Automatic PDE Loss Scaler Calculation
    # ====================================================================================
    if pde_loss_scaler == 'auto' and can_compute_pde and pde_weight > 0:
        print("--- Calibrating PDE loss scaler automatically ---")
        model.eval()
        total_data_loss_for_scaling = 0.0
        total_pde_loss_for_scaling = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)

                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                # Calculate data loss for this batch
                data_loss_batch = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))
                total_data_loss_for_scaling += data_loss_batch.item()

                # Calculate raw PDE loss for this batch
                pde_loss_batch_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                total_pde_loss_for_scaling += pde_loss_batch_raw.item()

        # Calculate the average losses
        avg_data_loss = total_data_loss_for_scaling / len(test_loader)
        avg_pde_loss = total_pde_loss_for_scaling / len(test_loader)

        # Compute the scaler
        if avg_pde_loss > 1e-8:  # Avoid division by zero
            pde_loss_scaler = avg_data_loss / avg_pde_loss
        else:
            pde_loss_scaler = 1.0  # Default to 1 if PDE loss is negligible

        print(f"Initial Avg Data Loss: {avg_data_loss:.6e}")
        print(f"Initial Avg Raw PDE Loss: {avg_pde_loss:.6e}")
        print(f"Calculated pde_loss_scaler: {pde_loss_scaler:.6f}")
        print("---------------------------------------------")

    elif not can_compute_pde or pde_weight == 0:
        pde_loss_scaler = 0.0  # No scaling needed if PDE is not used
    elif isinstance(pde_loss_scaler, str):  # Handle cases like 'auto' when PDE is off
        pde_loss_scaler = 1.0

    # --- Main Training Loop ---
    for ep in range(epochs):
        model.train()
        t1 = default_timer()

        # Initialize epoch-level accumulators
        train_mse_hybrid, train_l2_hybrid = 0.0, 0.0
        train_data, train_pde_scaled, train_pde_raw = 0.0, 0.0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            # --- Loss Calculation ---
            loss_data = myloss(out.flatten(start_dim=1), y.flatten(start_dim=1))
            mse_data = F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean')

            loss_pde_scaled = torch.tensor(0.0, device=device)
            loss_pde_raw = torch.tensor(0.0, device=device)

            if can_compute_pde and pde_weight > 0:
                loss_pde_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                loss_pde_scaled = loss_pde_raw * pde_loss_scaler

            # Combine losses
            loss_hybrid = (1.0 - pde_weight) * loss_data + pde_weight * loss_pde_scaled
            mse_hybrid = (1.0 - pde_weight) * mse_data + pde_weight * loss_pde_scaled

            loss_hybrid.backward()
            optimizer.step()
            scheduler.step()

            # --- Accumulate Metrics ---
            train_mse_hybrid += mse_hybrid.item()
            train_l2_hybrid += loss_hybrid.item()
            train_data += loss_data.item()
            train_pde_scaled += loss_pde_scaled.item()
            train_pde_raw += loss_pde_raw.item()

        # --- Evaluation ---
        model.eval()
        test_data, test_mse_data = 0.0, 0.0
        test_pde_loss_scaled, test_pde_loss_raw = 0.0, 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_data += myloss(out.flatten(start_dim=1), y.flatten(start_dim=1)).item()
                test_mse_data += F.mse_loss(out.flatten(start_dim=1), y.flatten(start_dim=1), reduction='mean').item()

                if can_compute_pde and pde_weight > 0:
                    loss_pde_test_raw = calculate_pde_residual(out, grid_info, epsilon, problem, device)
                    test_pde_loss_raw += loss_pde_test_raw.item()
                    test_pde_loss_scaled += (loss_pde_test_raw * pde_loss_scaler).item()

        # --- Normalize and Log Metrics ---
        # Note: We divide by len(loader) because item() gives the mean loss for the batch.
        # This computes the average of the batch means.

        # Averages for training set
        train_mse_hybrid /= len(train_loader)
        train_l2_hybrid /= len(train_loader)
        train_data /= len(train_loader)
        train_pde_scaled /= len(train_loader)
        train_pde_raw /= len(train_loader)

        # Averages for test set
        test_mse_data /= len(test_loader)
        test_data /= len(test_loader)
        test_pde_loss_scaled /= len(test_loader)
        test_pde_loss_raw /= len(test_loader)

        # Calculate final hybrid test losses from averages
        test_loss_hybrid = (1.0 - pde_weight) * test_data + pde_weight * test_pde_loss_scaled
        test_mse_hybrid = (1.0 - pde_weight) * test_mse_data + pde_weight * test_pde_loss_scaled

        # Append to logs
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

        t2 = default_timer()

        if ep == 0:
            print("--- Starting Training ---")
            print(f"PDE Loss Scaler is set to: {pde_loss_scaler:.6f}")
            print(
                "No. Epoch | Time (s)   | Train MSE Hyb  | Train L2 Hyb   | Test L2 Hyb    | Test L2 data   | Test PDE Scl.  | Test PDE Raw")
            print(
                "-------------------------------------------------------------------------------------------------------------------------")

        # CORRECTED PRINT STATEMENT:
        print(
            f"{ep:<9} | {t2 - t1:<10.4f} | {train_mse_hybrid:<14.6e} | {train_l2_hybrid:<14.6e} | {test_loss_hybrid:<14.6e} | {test_data:<14.6e} | {test_pde_loss_scaled:<14.6e} | {test_pde_loss_raw:<14.6e}")

    return model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log
'''