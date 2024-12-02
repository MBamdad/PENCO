"""
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""

import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

torch.manual_seed(0)
np.random.seed(0)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("Device name:")
print(device)
if torch.cuda.is_available():
    torch.cuda.set_device(0)


################################################################
# 3d fourier layers
################################################################

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

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channel):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic

        self.p = nn.Linear(in_channel,
                           self.width)  # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = self.q(x)
        #x = F.tanh(x)
        if classified:
            x = torch.sign(x)

        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


################################################################
# configs
################################################################
ntrain = 1000
ntest = 200

modes = 8
width = 20

batch_size = 10  #10

learning_rate = 0.0001  #0.001
weight_decay = 1e-4
epochs = 1000
iterations = epochs * (ntrain // batch_size)

normalized = True
classified = False
training = True

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
O = 64
S = O // sub
T_in = 1
T = 140

################################################################
# load data
################################################################
model_dir = './models'
model_filename = f'AC2D_model_S{S}_T{T_in}to_{T}_batch{batch_size}.pt'
model_path = os.path.join(model_dir, model_filename)

os.makedirs(model_dir, exist_ok=True)

TRAIN_PATH = 'data/AC2D_' + str(O) + '.mat'
TEST_PATH = TRAIN_PATH

dataset_name = 'AC2D'
parent_dir = './data/'
dataset_filename = parent_dir + dataset_name + '_S' + str(S) + '_from_' + str(O) + '_T_' + str(T_in) + '_to_' + str(
    T) + '.pt'

if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

if os.path.exists(dataset_filename):
    print("Found saved dataset at", dataset_filename)
    loaded_data = torch.load(dataset_filename)
    train_a = loaded_data['train_a']
    train_u = loaded_data['train_u']
    test_a = loaded_data['test_a']
    test_u = loaded_data['test_u']
else:
    reader = MatReader(TRAIN_PATH)
    train_a_mat = reader.read_field('phi')
    train_u_mat = reader.read_field('phi')
    train_a = train_a_mat[:ntrain, :T_in, ::sub, ::sub]
    train_u = train_u_mat[:ntrain, T_in:T + T_in, ::sub, ::sub]
    train_a = train_a.permute(0, 2, 3, 1)
    train_u = train_u.permute(0, 2, 3, 1)

    reader = MatReader(TEST_PATH)
    test_a_mat = reader.read_field('phi')
    test_u_mat = reader.read_field('phi')
    test_a = test_a_mat[-ntest:, :T_in, ::sub, ::sub]
    test_u = test_u_mat[-ntest:, T_in:T + T_in, ::sub, ::sub]
    test_a = test_a.permute(0, 2, 3, 1)
    test_u = test_u.permute(0, 2, 3, 1)

    torch.save({'train_a': train_a, 'train_u': train_u,
                'test_a': test_a, 'test_u': test_u}, dataset_filename)

print(train_u.shape)
print(test_u.shape)
print(train_a.shape)
print(test_a.shape)

if classified:
    train_u = torch.sign(train_u)
    test_u = torch.sign(test_u)
    train_a = torch.sign(train_a)
    test_a = torch.sign(test_a)

assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

if normalized:
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, T, 1])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2 - t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width, T_in + 3).to(device)
if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    print("No pre-trained model found. Initializing a new model.")
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)

train_mse_log = []
train_l2_log = []
test_l2_log = []

if normalized:
    y_normalizer.to(device)
    a_normalizer.to(device)

if training:
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x).view(batch_size, S, S, T)

            mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()

            if normalized:
                y = y_normalizer.decode(y)
                out = y_normalizer.decode(out)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()

            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x).view(batch_size, S, S, T)
                if normalized:
                    out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        train_mse_log.append(train_mse)
        train_l2_log.append(train_l2)
        test_l2_log.append(test_l2)

        t2 = default_timer()
        print(ep, t2 - t1, train_mse, train_l2, test_l2)

    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)

################################################################
# post-processing
################################################################
plt.semilogy(train_mse_log, label='Train MSE')
plt.semilogy(train_l2_log, label='Train L2')
plt.semilogy(test_l2_log, label='Test L2')
plt.legend()
plt.show()

inp = torch.zeros((ntest, S, S, T_in))
exact = torch.zeros(test_u.shape)
pred = torch.zeros(test_u.shape)
index = 0
test_l2_set = []
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).squeeze(-1)
        if normalized:
            out = y_normalizer.decode(out)
            x = a_normalizer.decode(x[:, :, :, 0, :])
        inp[index] = x
        exact[index] = y
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_l2_set.append(test_l2)
        print(index, test_l2)
        index = index + 1

test_l2_set = torch.tensor(test_l2_set)
test_l2_avg = torch.mean(test_l2_set)
test_l2_std = torch.std(test_l2_set)

print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())

# PLotting a random function from the test data generated by GRF
index = 50
T_index = 39
x_test_plot = np.linspace(0, 1, S).astype('float32')
y_test_plot = np.linspace(0, 1, S).astype('float32')
x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)

a_ind = inp[index, :, :, 0]
u_pred = pred[index, :, :, T_index]
u_exact = exact[index, :, :, T_index]

fig_font = "DejaVu Serif"
plt.rcParams["font.family"] = fig_font
plt.figure()
plt.contourf(x_test_plot, y_test_plot, a_ind, levels=1, cmap='hsv')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Input Function')
plt.show()

plt.figure()
plt.contourf(x_test_plot, y_test_plot, u_exact, levels=1, cmap='hsv')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Exact Value')
plt.show()

plt.figure()
plt.contourf(x_test_plot, y_test_plot, u_pred, levels=1, cmap='hsv')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Predicted Value')
plt.show()

plt.figure()
plt.contourf(x_test_plot, y_test_plot, u_pred - u_exact, levels=1, cmap='hsv')
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Error')
plt.show()

for T_index in range(pred.shape[-1]):
    u_pred = pred[index, :, :, T_index]
    plt.contourf(x_test_plot, y_test_plot, u_pred, levels=500, cmap='hsv')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Predicted Value at T_index = {T_index}')
    plt.show()
