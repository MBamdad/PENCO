import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

def get_grid_2d(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def get_grid_3d(shape, device):
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# Complex multiplication
def compl_mul2d(inp, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", inp, weights)


def compl_mul3d(inp, weights):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients --  1. Compute Fourier Transform
        # Now, instead of working with raw pixel/grid values, we work with frequency components\
        # Suppose the input x has shape (batch, in_channels, H, W)
        # x_ft has shape: batch×in_channels×H×(W/2+1)
        x_ft = torch.fft.rfft2(x) # 2D Fast Fourier Transform (FFT) --> v0 = F(v0)

        # Multiply relevant Fourier modes -- 2. Apply Spectral Convolution
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        # v1(k1,k2)= W(k1,k2).v0(k1,k2) --> W(k1,k2) are learnable parameters that control how much each frequency mode contributes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) # Apply Inverse Fourier Transform (iFFT) --> v1(x,y) = F^(-1)[v1(k1,k2)]
        return x


class SpectralConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

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
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, T=1, num_layers=2):
        """
        Initialize the MLP2d class.
        Parameters:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - mid_channels: Number of intermediate channels.
        - T: Number of blocks (default=1).
        - num_layers: Number of layers in each block (default=2).
        """
        super(MLP2d, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(T):
            self.layers.append(nn.Conv2d(in_channels, mid_channels, 1))
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Conv2d(mid_channels, mid_channels, 1))
            self.layers.append(nn.Conv2d(mid_channels, out_channels, 1))

    def forward(self, x, t=0):
        start = t * self.num_layers
        end = start + self.num_layers
        for i in range(start, end - 1):
            x = F.gelu(self.layers[i](x))
        x = self.layers[end - 1](x)
        return x


class MLP3d(MLP2d):
    def __init__(self, in_channels, out_channels, mid_channels, T=1, num_layers=2):
        super(MLP3d, self).__init__(in_channels, out_channels, mid_channels, T, num_layers)

        self.layers = nn.ModuleList()
        for _ in range(T):
            self.layers.append(nn.Conv3d(in_channels, mid_channels, 1))
            # After (3x3x3 kernel)
            #self.layers.append(nn.Conv3d(in_channels, mid_channels, 3, padding=1))  ## Changed from 1*1*1 to 3*3*3
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Conv3d(mid_channels, mid_channels, 1))
                #self.layers.append(nn.Conv3d(mid_channels, mid_channels, 3, padding=1))  ## Changed from 1 to 3
            self.layers.append(nn.Conv3d(mid_channels, out_channels, 1))
            #self.layers.append(nn.Conv3d(mid_channels, out_channels, 3, padding=1))  ## Changed from 1 to 3


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, width_q, T_in, T_out, n_layers):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 8  # pad the domain if input is non-periodic
        self.n_layers = n_layers

        self.p = nn.Linear(T_in + 2, self.width)  # We start with an input x == u(x,y) of shape (batch,x,y,c), We lift it to a higher-dimensional space using a linear layer
        # v0(x,y) = p(x=u)
        self.convs = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(n_layers)]) # 2D Fast Fourier Transform (FFT) --> v0 = F(v0)
        self.mlps = nn.ModuleList([MLP2d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(n_layers)]) # Pointwise convolution layers
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP2d(self.width, 1, self.width_q)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = get_grid_2d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            '''
            x1 = self.mlps[i](x1): Local Mixing Using MLP: Since the Fourier convolution captures global dependencies,
             we still need local interactions --> v_i+1 = sigma(W.vi  +  b), 
             which W and b are learnable parameters, and σ is the activation function.
            '''
            x2 = self.ws[i](x)
            '''
             x2 = self.ws[i](x): applies a pointwise convolution (1×1 convolution) to the input tensor x.
                self.ws is a list (nn.ModuleList) of 1×1 convolutional layers.
                Each self.ws[i] is a 2D convolution layer (nn.Conv2d) with a kernel size of 1x1.
                The purpose of these layers is to perform a linear transformation of the feature maps 
                without mixing spatial locations.

            '''
            x = x1 + x2 #  Merge Global and Local Representations
            x = F.gelu(x) if i < self.n_layers - 1 else x

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        '''
         Output Projection back to the desired shape using another MLP
        v_out = Q.v_final(x,y)
        Q is a learnable projection.

        '''
        x = x.permute(0, 2, 3, 1)
        #The final shape of x is (batch,x,y,1), which represents the predicted function value at each spatial location.
        return x


class TNO2d(FNO2d):
    def __init__(self, modes1, modes2, width, width_q, width_h, T_in, T_out, n_layers, n_layers_q=2, n_layers_h=4):
        super(TNO2d, self).__init__(modes1, modes2, width, width_q, T_in, T_out, n_layers)
        '''
         TNO2d extends FNO2d. It introduces temporal modeling by adding two MLP layers:
        self.q → projects the Fourier features to output over time.
        self.h → handles temporal dependencies between consecutive time steps.
        New parameters added:
        width_h → controls temporal memory features.
        n_layers_q → depth of self.q (output MLP).
        n_layers_h → depth of self.h (temporal evolution MLP).
        '''
        self.width_h = width_h
        #self.q = MLP2d(self.width, 1, self.width, T_out) # for AC
        #self.q2 = MLP2d(1, 1, self.width // 4, T_out - 1)
        #self.q = MLP2d(self.width, 1, 2 * self.width, T_out)  # for CH3D
        #self.q2 = MLP2d(1, 1, self.width, T_out - 1)
        self.q = MLP2d(self.width, 1, self.width_q, T_out, n_layers_q)  # for CHNL
        self.h = MLP2d(1, 1, self.width_h, T_out - 1, n_layers_h)

    def forward(self, x):
        grid = get_grid_2d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # a(x) or= x : Input function (e.g., initial condition for a PDE)
        x = self.p(x) # 	Lifts input to a high-dimensional space
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x # x′=GELU(FourierConv(x)+MLP(x)+PointwiseConv(x)

        # x = x[..., :-self.padding, :-self.padding]
        '''
         Temporal Evolution Loop
        Initial time step prediction:
        Uses self.q(x) to generate the first time step.
        Stores result in X[..., 0]
        '''
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)
        xt = self.q(x)
        X[..., 0] = xt.permute(0, 2, 3, 1).squeeze(-1)

        for t in range(1, self.T_out):
            x1 = self.q(x, t) # Predicts the next step using Fourier features.  # Q_n∘(W_L+ K_L )∘...∘P(a(x)), Projects final Fourier features to outpu
            x2 = self.h(xt, t - 1) # Uses previous output (xt) to refine the next state. # H_n∘G_θ (x,t_(n-1) )(a(x)), Models dependency on past states
            xt = x1 + x2 #  Solution at time t_n : x_t = G_θ (x,t_n )(a(x))
            X[..., t] = xt.permute(0, 2, 3, 1).squeeze(-1)
            '''
             Uses previous output (xt) to refine the next state.
            Combines both predictions --> x_t=MLP_q(x)+MLP_h[(x t−1)]
            Stores result in X[..., t]
            '''
        return X


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, width_q, T_in, T_out, n_layers, n_layers_q=2, n_layers_h=2):
        super(FNO3d, self).__init__()

        """
        The FNO3d class is a deep learning model designed for solving spatiotemporal problems. 
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 time_steps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 time_steps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 6  # pad the domain if input is non-periodic
        self.n_layers = n_layers

        self.p = nn.Linear(self.T_in + 3, self.width)  # Lifting Layer: input channel is 12: the solution of the first 10 time_steps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.convs = nn.ModuleList(
            [SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([MLP3d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])
        #self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 3, padding=1) for _ in range(n_layers)])  ## kernel changed
        #self.q = MLP3d(self.width, 1, self.width)  # output channel is 1: u(x, y)
        self.q = MLP3d(self.width, 1, self.width_q)  # output channel is 1: u(x, y)

    def forward(self, x):
        #x = x.unsqueeze(3).repeat([1, 1, 1, self.T_out, 1])
        grid = get_grid_3d(x.shape, x.device)
        #print(' x shape:', x.shape)
        x = torch.cat((x, grid), dim=-1)
        #print(' x shape after cat:', x.shape)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        #x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        #print(' x shape after permute:', x.shape)
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        #x = x[..., :-self.padding]
        x = self.q(x)
        #x = x.permute(0, 2, 3, 4, 1)[..., 0]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 4, 1)
        #print('FNO3d return x shape:', x.shape)
        return x


class TNO3d(FNO3d):
    def __init__(self, modes1, modes2, modes3, width, width_q, width_h, T_in, T_out, n_layers):
        super(TNO3d, self).__init__(modes1, modes2, modes3, width, width_q, T_in, T_out, n_layers)
        """
        The super() function calls the parent class (FNO3d) constructor to initialize 
        the parameters that are inherited from the parent class.
        input: the initial condition and locations (a(x, y, z), x, y, z)
        input shape: (batchsize, x=s, y=s, z=s, c=4)
        output: the solution 
        output shape: (batchsize, x=s, y=s, z=s, t=T)
        """
        self.width_h = width_h

        #self.q = MLP3d(self.width, 1, self.width, T_out)
        #self.q2 = MLP3d(1, 1, self.width // 4, T_out - 1)
        self.q = MLP3d(self.width, 1, self.width_q, T_out)
        self.h = MLP3d(1, 1, self.width_h, T_out - 1)

    def forward(self, x):
        grid = get_grid_3d(x.shape, x.device)
        #print('x shape: ',x.shape)
        #print('grid shape: ', grid.shape)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        # x = x[..., :-self.padding, :-self.padding]
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)
        xt = self.q(x)
        X[..., 0] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)
        for t in range(1, self.T_out):
            x1 = self.q(x, t)
            x2 = self.h(xt, t - 1)
            xt = x1 + x2
            X[..., t] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)

        #print('shape X for model TNO: ', X.shape)
        return X


def get_grid_3D(shape, device):
    batchsize, size_x, size_y, size_z, _ = shape  # Note: last dim is channels, not time
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # Returns (batch, x, y, z, 3)


class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4_internal, width, width_q, T_in_channels, n_layers):
        super(FNO4d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4_internal
        self.width = width
        self.width_q = width_q
        self.T_in = T_in_channels
        self.n_layers = n_layers
        self.padding = 6

        # Input is (x,y,z) + time channels (t_in_channels) + 3 spatial coordinates
        self.p = nn.Linear(self.T_in + 3, self.width)  # +3 for (x,y,z) coordinates

        self.convs = nn.ModuleList([
            SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            for _ in range(n_layers)
        ])
        self.mlps = nn.ModuleList([MLP3d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])
        self.q = MLP3d(self.width, 1, self.width_q)  # Output channel is 1

    def forward(self, x):
        # Input shape: (batch, x, y, z, t_in_channels)
        grid = get_grid_3D(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # Now both are 5D: (batch, x, y, z, t_in_channels + 3)
        x = self.p(x)  # Lift to higher dimension
        x = x.permute(0, 4, 1, 2, 3)  # (batch, channels, x, y, z)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)  # (batch, x, y, z, 1)
        return x

#### FFNO4d    #####
####

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Helpers for complex multiplication (einsum)
# -----------------------------
def compl_mul1d_x(inp, weights):
    # inp: (B, Cin, Kx, Y, Z), w: (Cin, Cout, Kx) -> (B, Cout, Kx, Y, Z)
    return torch.einsum("bcxyz,cok->boxyz", inp, weights)

def compl_mul1d_y(inp, weights):
    # inp: (B, Cin, X, Ky, Z), w: (Cin, Cout, Ky) -> (B, Cout, X, Ky, Z)
    return torch.einsum("bcxyz,coy->boxyz", inp, weights)

def compl_mul1d_z(inp, weights):
    # inp: (B, Cin, X, Y, Kz), w: (Cin, Cout, Kz) -> (B, Cout, X, Y, Kz)
    return torch.einsum("bcxyz,coz->boxyz", inp, weights)

def compl_mul2d_xy(inp, weights):
    # inp: (B, Cin, Kx, Ky, Z)
    # w  : (Cin, Cout, Kx, Ky)
    # out: (B, Cout, Kx, Ky, Z)
    return torch.einsum("b c x y z, c o x y -> b o x y z", inp, weights)

def compl_mul2d_xz(inp, weights):
    # inp: (B, Cin, Kx, Y, Kz)
    # w  : (Cin, Cout, Kx, Kz)
    # out: (B, Cout, Kx, Y, Kz)
    return torch.einsum("b c x y z, c o x z -> b o x y z", inp, weights)

def compl_mul2d_yz(inp, weights):
    # inp: (B, Cin, X, Ky, Kz)
    # w  : (Cin, Cout, Ky, Kz)
    # out: (B, Cout, X, Ky, Kz)
    return torch.einsum("b c x y z, c o y z -> b o x y z", inp, weights)


class EnhancedFactorizedSpectralConv3d(nn.Module):
    """
    Enhanced FFNO-style operator in 3D:
      K(x) = Kx + Ky + Kz + Kxy + Kxz + Kyz

    - Kx,Ky,Kz: 1D factorized spectral mixing per axis
    - Kxy,Kxz,Kyz: pairwise spectral couplings (restores cross-dim interactions)
    """

    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z, use_pairwise=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_pairwise = use_pairwise

        scale = 1.0 / (in_channels * out_channels)

        # 1D (x,y,z)
        self.weights_x = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, dtype=torch.cfloat))
        self.weights_y = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_y, dtype=torch.cfloat))
        self.weights_z = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_z, dtype=torch.cfloat))

        # 2D pairwise (xy,xz,yz)
        if self.use_pairwise:
            self.weights_xy = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
            self.weights_xz = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_z, dtype=torch.cfloat))
            self.weights_yz = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_y, modes_z, dtype=torch.cfloat))

    def forward(self, x):
        """
        x: (B, Cin, X, Y, Z) real
        returns a dict of components, each (B, Cout, X, Y, Z)
        """
        B, Cin, X, Y, Z = x.shape
        device = x.device

        comps = {}

        # ---- 1D: X ----
        x_ft = torch.fft.rfft(x, dim=-3)
        out_ft = torch.zeros(B, self.out_channels, X // 2 + 1, Y, Z, dtype=torch.cfloat, device=device)
        mx = min(self.modes_x, X // 2 + 1)
        out_ft[:, :, :mx, :, :] = compl_mul1d_x(x_ft[:, :, :mx, :, :], self.weights_x[:, :, :mx])
        comps["x"] = torch.fft.irfft(out_ft, n=X, dim=-3)

        # ---- 1D: Y ----
        y_ft = torch.fft.rfft(x, dim=-2)
        out_ft = torch.zeros(B, self.out_channels, X, Y // 2 + 1, Z, dtype=torch.cfloat, device=device)
        my = min(self.modes_y, Y // 2 + 1)
        out_ft[:, :, :, :my, :] = compl_mul1d_y(y_ft[:, :, :, :my, :], self.weights_y[:, :, :my])
        comps["y"] = torch.fft.irfft(out_ft, n=Y, dim=-2)

        # ---- 1D: Z ----
        z_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=device)
        mz = min(self.modes_z, Z // 2 + 1)
        out_ft[:, :, :, :, :mz] = compl_mul1d_z(z_ft[:, :, :, :, :mz], self.weights_z[:, :, :mz])
        comps["z"] = torch.fft.irfft(out_ft, n=Z, dim=-1)

        if not self.use_pairwise:
            return comps

        # ---- 2D: XY ----
        xy_ft = torch.fft.rfftn(x, dim=[-3, -2])
        out_ft = torch.zeros(B, self.out_channels, X, Y // 2 + 1, Z, dtype=torch.cfloat, device=device)

        mx2 = min(self.modes_x, X)
        my2 = min(self.modes_y, Y // 2 + 1)
        out_ft[:, :, :mx2, :my2, :] = compl_mul2d_xy(
            xy_ft[:, :, :mx2, :my2, :], self.weights_xy[:, :, :mx2, :my2]
        )
        out_ft[:, :, -mx2:, :my2, :] = compl_mul2d_xy(
            xy_ft[:, :, -mx2:, :my2, :], self.weights_xy[:, :, :mx2, :my2]
        )
        comps["xy"] = torch.fft.irfftn(out_ft, s=(X, Y), dim=[-3, -2])

        # ---- 2D: XZ ----
        xz_ft = torch.fft.rfftn(x, dim=[-3, -1])
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=device)

        mx2 = min(self.modes_x, X)
        mz2 = min(self.modes_z, Z // 2 + 1)

        out_ft[:, :, :mx2, :, :mz2] = compl_mul2d_xz(
            xz_ft[:, :, :mx2, :, :mz2], self.weights_xz[:, :, :mx2, :mz2]
        )
        out_ft[:, :, -mx2:, :, :mz2] = compl_mul2d_xz(
            xz_ft[:, :, -mx2:, :, :mz2], self.weights_xz[:, :, :mx2, :mz2]
        )
        comps["xz"] = torch.fft.irfftn(out_ft, s=(X, Z), dim=[-3, -1])

        # ---- 2D: YZ ----
        yz_ft = torch.fft.rfftn(x, dim=[-2, -1])
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=device)

        my2 = min(self.modes_y, Y)
        mz2 = min(self.modes_z, Z // 2 + 1)

        out_ft[:, :, :, :my2, :mz2] = compl_mul2d_yz(
            yz_ft[:, :, :, :my2, :mz2], self.weights_yz[:, :, :my2, :mz2]
        )
        out_ft[:, :, :, -my2:, :mz2] = compl_mul2d_yz(
            yz_ft[:, :, :, -my2:, :mz2], self.weights_yz[:, :, :my2, :mz2]
        )
        comps["yz"] = torch.fft.irfftn(out_ft, s=(Y, Z), dim=[-2, -1])

        return comps


class FFNOBlock3d(nn.Module):
    """
    Improved FFNO block:
      - Enhanced spectral operator: 1D + pairwise couplings   (Option B)
      - Local 3x3x3 conv path (stencil-like bias)             (Option C)
      - Residual-after-nonlinearity (F-FNO stability trick)
      - Channel-mixing MLP (1x1x1 convs)
    """

    def __init__(self, width, modes, expansion=2, use_pairwise=True, use_local_conv=True):
        super().__init__()

        self.K = EnhancedFactorizedSpectralConv3d(
            in_channels=width,
            out_channels=width,
            modes_x=modes,
            modes_y=modes,
            modes_z=modes,
            use_pairwise=use_pairwise
        )
        # Learnable scalar gates (start near equal contribution)
        # order: x, y, z, xy, xz, yz, local
        self.g_logits = nn.Parameter(torch.zeros(7))


        self.use_local_conv = use_local_conv
        if use_local_conv:
            self.local = nn.Conv3d(width, width, kernel_size=3, padding=1)
        else:
            self.local = None

        #self.norm = nn.InstanceNorm3d(width)
        self.norm = nn.GroupNorm(1, width)  # behaves like LayerNorm over channels

        hidden = expansion * width
        self.w1 = nn.Conv3d(width, hidden, kernel_size=1)
        self.w2 = nn.Conv3d(hidden, width, kernel_size=1)

    def forward(self, x):
        # x: (B, width, X, Y, Z)
        comps = self.K(x)  # dict of components

        g = F.softmax(self.g_logits, dim=0)
        k = (
                g[0] * comps["x"] +
                g[1] * comps["y"] +
                g[2] * comps["z"] +
                g[3] * comps["xy"] +
                g[4] * comps["xz"] +
                g[5] * comps["yz"]
        )
        if self.local is not None:
            k = k + g[6] * self.local(x)

        k = self.norm(k)

        y = self.w2(F.gelu(self.w1(k)))
        y = F.gelu(y)

        return x + y


class FFNO4d(nn.Module):
    """
    Full FFNO backbone for your pipeline.

    Input : (B, X, Y, Z, T_in_channels)
    Output: (B, X, Y, Z, 1)

    Compatible with your main.py selection:
      model = FFNO4d(modes=config.MODES, width=config.WIDTH, width_q=config.WIDTH_Q,
                    T_in_channels=config.T_IN_CHANNELS, n_layers=config.N_LAYERS)
    """

    def __init__(self, modes, width, width_q, T_in_channels, n_layers,
                 expansion=2, use_pairwise=True, use_local_conv=True):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in_channels
        self.n_layers = n_layers

        # Lift (T_in + 3 coords) -> width
        self.p = nn.Linear(self.T_in + 3, self.width)

        # Stack operator blocks
        self.blocks = nn.ModuleList([
            FFNOBlock3d(
                width=self.width,
                modes=self.modes,
                expansion=expansion,
                use_pairwise=use_pairwise,
                use_local_conv=use_local_conv
            )
            for _ in range(self.n_layers)
        ])

        # Output head: simple and robust (does not depend on your MLP3d signature)
        self.proj1 = nn.Conv3d(self.width, self.width_q, kernel_size=1)
        self.proj2 = nn.Conv3d(self.width_q, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in_channels)
        grid = get_grid_3D(x.shape, x.device)    # must exist in your networks.py
        x = torch.cat((x, grid), dim=-1)         # (B,X,Y,Z,T_in+3)

        x = self.p(x)                            # (B,X,Y,Z,width)
        x = x.permute(0, 4, 1, 2, 3)             # (B,width,X,Y,Z)

        for blk in self.blocks:
            x = blk(x)

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)                        # (B,1,X,Y,Z)
        x = x.permute(0, 2, 3, 4, 1)             # (B,X,Y,Z,1)
        return x


##################################
############# MHNO with FFNO backbone

class FFNO3d(nn.Module):
    """
    FFNO backbone (like FNO3d but operator is FFNO blocks).
    This returns a single-step prediction (B, X, Y, Z, 1) like FNO3d does.

    Input:  (B, X, Y, Z, T_in)
    Output: (B, X, Y, Z, 1)
    """
    def __init__(self, modes, width, width_q, T_in, n_layers,
                 expansion=2, use_pairwise=True, use_local_conv=True):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.n_layers = n_layers

        # Lift (T_in + 3 coords) -> width
        self.p = nn.Linear(self.T_in + 3, self.width)

        # FFNO operator blocks
        self.blocks = nn.ModuleList([
            FFNOBlock3d(
                width=self.width,
                modes=self.modes,
                expansion=expansion,
                use_pairwise=use_pairwise,
                use_local_conv=use_local_conv
            )
            for _ in range(self.n_layers)
        ])

        # Projection head to 1 channel (single-step)
        self.proj1 = nn.Conv3d(self.width, self.width_q, kernel_size=1)
        self.proj2 = nn.Conv3d(self.width_q, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)
        grid = get_grid_3d(x.shape, x.device)     # (B,X,Y,Z,3)
        x = torch.cat((x, grid), dim=-1)          # (B,X,Y,Z,T_in+3)

        x = self.p(x)                             # (B,X,Y,Z,width)
        x = x.permute(0, 4, 1, 2, 3)              # (B,width,X,Y,Z)

        for blk in self.blocks:
            x = blk(x)

        y = F.gelu(self.proj1(x))
        y = self.proj2(y)                         # (B,1,X,Y,Z)
        y = y.permute(0, 2, 3, 4, 1)              # (B,X,Y,Z,1)
        return y


class MHNO_FFNO(FFNO3d):
    """
    MHNO / TNO-style model but with FFNO backbone.
    Mirrors your TNO3d logic:
      - backbone features from FFNO blocks
      - q(x,t): time-dependent projection from features to state
      - h(xt, t-1): history correction using previous predicted state

    Input:  (B, X, Y, Z, T_in)   (e.g., T_in=4 past frames)
    Output: (B, X, Y, Z, T_out)  (autoregressive rollout length)
    """
    def __init__(self, modes, width, width_q, width_h, T_in, T_out, n_layers,
                 expansion=2, use_pairwise=True, use_local_conv=True,
                 n_layers_q=2, n_layers_h=2):
        super().__init__(modes=modes, width=width, width_q=width_q, T_in=T_in, n_layers=n_layers,
                         expansion=expansion, use_pairwise=use_pairwise, use_local_conv=use_local_conv)

        self.T_out = T_out
        self.width_h = width_h

        # Replace the single-step proj head with time-conditioned heads like TNO3d:
        # q: maps backbone features -> next state, conditioned on time index t
        # h: maps previous predicted state -> correction, conditioned on (t-1)

        # IMPORTANT: your existing MLP3d supports "T blocks" via forward(x, t=...)
        # We'll use it exactly like your TNO3d does.
        self.q_time = MLP3d(self.width, 1, self.width_q, T_out, num_layers=n_layers_q)   # (B,1,X,Y,Z) per t
        self.h_time = MLP3d(1, 1, self.width_h, max(T_out - 1, 1), num_layers=n_layers_h)

        # Keep the backbone blocks from FFNO3d (self.blocks) and lifting layer self.p.

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)
        grid = get_grid_3d(x.shape, x.device)     # (B,X,Y,Z,3)
        x = torch.cat((x, grid), dim=-1)          # (B,X,Y,Z,T_in+3)

        x = self.p(x)                             # (B,X,Y,Z,width)
        x = x.permute(0, 4, 1, 2, 3)              # (B,width,X,Y,Z)

        # Backbone operator stack (FFNO)
        for blk in self.blocks:
            x = blk(x)

        # Autoregressive rollout like TNO3d
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)  # (B,X,Y,Z,T_out)

        # First step uses q at t=0
        xt = self.q_time(x, t=0)                                  # (B,1,X,Y,Z)
        X[..., 0] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)          # (B,X,Y,Z)

        # Next steps
        for t in range(1, self.T_out):
            x1 = self.q_time(x, t=t)                              # (B,1,X,Y,Z)
            x2 = self.h_time(xt, t=t-1)                            # (B,1,X,Y,Z)
            xt = x1 + x2
            X[..., t] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)

        return X
