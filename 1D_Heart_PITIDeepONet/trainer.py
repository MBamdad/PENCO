# trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Heat1DConfig
from networks import DeepONet1DHeat
from utilities import relative_l2, make_integrator, residual_pointwise, make_uniform_grid

class PITITrainer:
    def __init__(self, cfg: Heat1DConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, cfg.dtype)

        self.model = DeepONet1DHeat(
            Nx=cfg.Nx,
            hidden_dim=cfg.hidden_dim,
            branch_layers=cfg.branch_layers,
            trunk_layers=cfg.trunk_layers,
            activation=cfg.activation,
        ).to(self.device).to(self.dtype)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.base_lr,
            betas=cfg.adam_betas,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay_gamma
        )
        self.mse = nn.MSELoss(reduction="mean")

        # prebuild uniform grid (for interpolation helper)
        self.x_grid = make_uniform_grid(cfg.x0, cfg.x1, cfg.Nx, device=self.device, dtype=cfg.dtype)

    def _sample_points(self, B: int):
        cfg = self.cfg
        dtype = self.dtype
        x_r = torch.rand(B, cfg.Nr_per_profile, 1, device=self.device, dtype=dtype)
        x_s = torch.rand(B, cfg.Ns_per_profile, 1, device=self.device, dtype=dtype)
        x_c = torch.rand(B, cfg.Nc_per_profile, 1, device=self.device, dtype=dtype)
        Nb = cfg.Nb_per_profile
        xb = torch.zeros(B, Nb, 1, device=self.device, dtype=dtype)
        if Nb >= 1: xb[:, 0, 0] = 0.0
        if Nb >= 2: xb[:, 1, 0] = 1.0
        if Nb >  2: xb[:, 2:, 0] = torch.rand(B, Nb - 2, device=self.device, dtype=dtype)
        return {"x_r": x_r, "x_s": x_s, "x_c": x_c, "x_b": xb}

    def _gather_sensors(self, u_batch, x_grid, x_query):
        """
        Interpolate u on uniform grid to query points.
        """
        B, M, _ = x_query.shape
        Nx = x_grid.shape[0]
        xi = x_query.squeeze(-1) * (Nx - 1)
        i0 = torch.clamp(xi.floor().long(), 0, Nx - 2)
        i1 = i0 + 1
        w = (xi - i0.to(xi.dtype))
        u0 = torch.gather(u_batch, 1, i0)
        u1 = torch.gather(u_batch, 1, i1)
        u_interp = (1 - w) * u0 + w * u1
        return u_interp

    def _pde_residual(self, u_sensors, x_points):
        cfg = self.cfg
        B, M, _ = x_points.shape
        x = x_points.detach().clone().requires_grad_(True)
        t_ph = torch.zeros_like(x, requires_grad=True)
        coords = torch.cat([x, t_ph], dim=-1)
        u_hat, ut_hat = self.model(u_sensors, coords)  # (B,M)
        grad_u = torch.autograd.grad(u_hat.sum(), x, create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(grad_u.sum(), x, create_graph=True, retain_graph=True)[0].squeeze(-1)
        r = ut_hat - cfg.alpha * u_xx
        return r

    def _consistency(self, u_sensors, x_points):
        x = x_points.detach().clone()
        t_ph = torch.zeros_like(x, requires_grad=True)
        coords = torch.cat([x, t_ph], dim=-1)
        u_hat, ut_hat = self.model(u_sensors, coords)
        du_dt = torch.autograd.grad(u_hat.sum(), t_ph, create_graph=False, retain_graph=False)[0].squeeze(-1)
        return ut_hat, du_dt

    def train_epoch(self, loader, x_grid):
        cfg = self.cfg
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            assert batch["mode"][0] == "pi_profile"
            u_sensors = torch.stack([b.to(self.device).to(self.dtype) for b in batch["u"]], dim=0)  # (B,Nx)

            B = u_sensors.shape[0]
            pts = self._sample_points(B)
            x_r, x_s, x_c, x_b = pts["x_r"], pts["x_s"], pts["x_c"], pts["x_b"]

            t0_r = torch.zeros(B, x_r.shape[1], 1, device=self.device, dtype=self.dtype)
            t0_s = torch.zeros(B, x_s.shape[1], 1, device=self.device, dtype=self.dtype)
            t0_b = torch.zeros(B, x_b.shape[1], 1, device=self.device, dtype=self.dtype)

            coords_r = torch.cat([x_r, t0_r], dim=-1)
            coords_s = torch.cat([x_s, t0_s], dim=-1)
            coords_b = torch.cat([x_b, t0_b], dim=-1)

            u_hat_s, ut_hat_s = self.model(u_sensors, coords_s)
            u_true_s = self._gather_sensors(u_sensors, x_grid, x_s)

            L_R = torch.mean((u_hat_s - u_true_s) ** 2)
            r = self._pde_residual(u_sensors, x_r)
            L_PDE = torch.mean(r ** 2)
            u_hat_b = self.model.forward_only_u(u_sensors, coords_b)
            L_BC = torch.mean(u_hat_b ** 2)
            ut_hat_c, du_dt_c = self._consistency(u_sensors, x_c)
            L_C = torch.mean((ut_hat_c - du_dt_c) ** 2)

            loss = (cfg.lambda_PDE * L_PDE +
                    cfg.lambda_R * L_R +
                    cfg.lambda_BC * L_BC +
                    cfg.lambda_C * L_C)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def infer_operator(self):
        def G(u_grid, x_grid):
            self.model.eval()
            u_batch = u_grid.unsqueeze(0)
            x = x_grid.unsqueeze(0).unsqueeze(-1)
            t0 = torch.zeros_like(x)
            coords = torch.cat([x, t0], dim=-1)
            _, ut = self.model(u_batch, coords)
            return ut.squeeze(0)
        return G

    @torch.no_grad()
    def roll_and_collect(self, u0, T_final, dt, scheme="RK4"):
        """
        Roll the learned operator from u0 up to T_final with step dt.
        Returns times (S+1,), states (S+1, Nx).
        """
        integrator = make_integrator(scheme)
        x_grid = self.x_grid
        G = self.infer_operator()
        steps = int(round(T_final / dt))
        U = [u0.clone()]
        u = u0.clone()
        if scheme.lower() == "abm2":
            # bootstrap u_{-1} via one Euler step
            u_nm1 = u.clone()
            u = u + dt * G(u, x_grid)
        for k in range(steps):
            if scheme.lower() == "abm2":
                u_next = integrator(G, u_nm1, u, dt, x_grid, self.device)
                u_nm1 = u.clone()
                u = u_next
            else:
                u = integrator(G, u, dt, x_grid, self.device)
            U.append(u.clone())
        t = torch.linspace(0.0, dt * steps, steps + 1, device=u0.device, dtype=u0.dtype)
        return t, torch.stack(U, dim=0)
