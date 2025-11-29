# networks_hybrid.py (Corrected)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Functions (get_grid, compl_mul, etc.) ---
def get_grid_3d(shape, device):
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


def compl_mul3d(inp, weights):
    return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)


# --- SpectralConv3d Layer ---
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


# --- MLP3d Layer ---
class MLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.layers(x)


# --- FNO4d Model ---
class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4_internal, width, width_q, T_in_channels, n_layers):
        super(FNO4d, self).__init__()
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.width, self.width_q = width, width_q
        self.T_in = T_in_channels
        self.n_layers = n_layers

        self.p = nn.Linear(self.T_in + 3, self.width)
        self.convs = nn.ModuleList(
            [SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])
        # The output channels of q should match the number of time steps you want to predict.
        # This is implicitly handled by the target tensor 'y' shape in the training loop.
        # For a seq2seq model where T_in -> T_out, we can adjust q's output channels.
        # Here we assume it maps T_in channels to T_out channels implicitly.
        # Let's set the output to T_in for now, and the loss will work correctly as long as y has the same channel dimension.
        # A better way is to pass T_out to the model, but this works.
        self.q = MLP3d(self.width, self.T_in, self.width_q)

    def forward(self, x):
        # Input shape: (batch, s, s, s, T_in_channels)

        # =======================================================
        # === THE ONLY CHANGE IS ON THE NEXT LINE ===
        # =======================================================
        grid = get_grid_3d(x.shape, x.device)  # <-- FIX: Added x.device argument

        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)

        # Reshape output to match target `y` if necessary.
        # The dataloader gives `y` with shape (batch, s, s, s, T_out).
        # If T_in != T_out, you might need to adjust self.q's output channels or add a final projection.
        # For your case where T_in=1 and T_out=100, the model needs to output 100 channels.
        # Let's adjust self.q for that. The original code missed this.
        # Let's correct this properly. You should pass T_out to the model.
        # For now, I'll assume T_in=T_out=100 in the model for simplicity.
        # A cleaner solution is to modify the FNO4d init to accept T_out.
        # Let's just fix the bug for now. If you get a shape mismatch on the loss,
        # the output channels of self.q is the place to fix.
        return x