# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_mlp(in_dim, hidden_dim, layers, activation="tanh", out_dim=None):
    acts = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    Act = acts[activation]
    modules = []
    dim = in_dim
    for _ in range(layers):
        modules += [nn.Linear(dim, hidden_dim), Act()]
        dim = hidden_dim
    if out_dim is not None:
        modules += [nn.Linear(dim, out_dim)]
    return nn.Sequential(*modules)

class DeepONet1DHeat(nn.Module):
    """
    Unstacked DeepONet with shared branch & trunk; two heads for outputs:
      - head_u: reconstruction \hat u(x)
      - head_ut: time derivative \hat u_t(x)
    Branch input: sensor values of u^n on Nx grid (size 128).
    Trunk input: (x, t_phantom), with t_phantom=0 during training (kept grad-enabled).
    Activation: tanh (per paper).
    """
    def __init__(self, Nx, hidden_dim=256, branch_layers=8, trunk_layers=10, activation="tanh"):
        super().__init__()
        self.Nx = Nx

        # Branch encodes the function u^n sampled at Nx sensors
        self.branch = make_mlp(Nx, hidden_dim, branch_layers, activation=activation)

        # Trunk encodes coordinates (x, t)
        self.trunk = make_mlp(2, hidden_dim, trunk_layers, activation=activation)

        # Two heads combine branch & trunk features via dot-product-like interaction
        # We implement as elementwise product then a linear readout.
        self.readout_u = nn.Linear(hidden_dim, 1)
        self.readout_ut = nn.Linear(hidden_dim, 1)

    def forward(self, u_sensors, coords):
        """
        u_sensors: (B, Nx)
        coords: (B, M, 2) with columns [x, t_phantom]
        Returns:
          u_hat: (B, M)
          ut_hat: (B, M)
        """
        B, M, _ = coords.shape
        # Encode branch once per profile
        b = self.branch(u_sensors)                # (B, H)
        # Encode trunk per coordinate
        coords_flat = coords.reshape(B * M, 2)
        t = self.trunk(coords_flat)               # (B*M, H)
        # Match dimensions
        b_rep = b.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)  # (B*M, H)
        # Interaction
        phi = b_rep * t                           # (B*M, H)
        u_hat = self.readout_u(phi).reshape(B, M)   # (B, M)
        ut_hat = self.readout_ut(phi).reshape(B, M) # (B, M)
        return u_hat, ut_hat

    def forward_only_u(self, u_sensors, coords):
        u_hat, _ = self.forward(u_sensors, coords)
        return u_hat

    def forward_only_ut(self, u_sensors, coords):
        _, ut_hat = self.forward(u_sensors, coords)
        return ut_hat
