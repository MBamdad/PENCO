"""
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
import os
import matplotlib.pyplot as plt
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print("Device name:")
print(device)
if torch.cuda.is_available():
    torch.cuda.set_device(0)


################################################################
# fourier layer
################################################################
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

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, T=1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(T):
            self.layers.append(nn.Conv2d(in_channels, mid_channels, 1))
            self.layers.append(nn.Conv2d(mid_channels, out_channels, 1))

    def forward(self, x, t=1):
        x = self.layers[2 * (t - 1)](x)
        x = F.gelu(x)
        x = self.layers[2 * (t - 1) + 1](x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, T=1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.T = T

        self.p = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4, T)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

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

        x = x[..., :-self.padding, :-self.padding]

        X = torch.zeros(grid.shape[0:-1]).unsqueeze(-1).expand(-1, -1, -1, self.T).to(device)
        for t in range(self.T):
            X[..., t] = self.q(x, t).permute(0, 2, 3, 1).squeeze(-1)
        return X

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################
ntrain = 1000
ntest = 100


batch_size = 10#40
learning_rate = 0.001
weight_decay = 1e-4
epochs = 100#500
iterations = epochs * (ntrain // batch_size)

modes = 8
width = 20#32

s = 64
T = 140

normalized = True
training = True
################################################################
# load data and data normalization
################################################################
problem = 'AC2D'

model_dir = './models'
model_name = f'{problem}_model_S{s}_T{T}_batch{batch_size}.pt'
model_path = os.path.join(model_dir, model_name)
os.makedirs(model_dir, exist_ok=True)

DATA_PATH = f'data/{problem}_{s}.mat'

parent_dir = './data/'
dataset_file = parent_dir + problem + '_S' + str(s) + '_T_' + str(T) + '.pt'

if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

if os.path.exists(dataset_file):
    print("Found saved dataset at", dataset_file)
    loaded_data = torch.load(dataset_file)
    train_a = loaded_data['train_a']
    train_u = loaded_data['train_u']
    test_a = loaded_data['test_a']
    test_u = loaded_data['test_u']
else:
    reader = MatReader(DATA_PATH)
    train_a_mat = reader.read_field('phi')
    train_u_mat = reader.read_field('phi')
    train_a = train_a_mat[:ntrain, 0, :s, :s]
    train_u = train_u_mat[:ntrain, 1:T+1, :s, :s]
    train_u = train_u.permute(0, 2, 3, 1)
    train_a = train_a.reshape(ntrain, s, s, 1)

    test_a = train_a_mat[-ntest:, 0, :s, :s]
    test_u = train_a_mat[-ntest:, 1:T+1, :s, :s]
    test_u = test_u.permute(0, 2, 3, 1)
    test_a = test_a.reshape(ntest, s, s, 1)

    torch.save({'train_a': train_a, 'train_u': train_u,
                'test_a': test_a, 'test_u': test_u}, dataset_file)

print(train_u.shape)
print(test_u.shape)
print(train_a.shape)
print(test_a.shape)

if normalized:
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
    train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
    test_a, test_u), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width, T).to(device)
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
            out = model(x)
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
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

inp = torch.zeros((ntest, s, s))
exact = torch.zeros(test_u.shape)
pred = torch.zeros(test_u.shape)
index = 0
test_l2_set = []
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x)
        if normalized:
            out = y_normalizer.decode(out)
            x = a_normalizer.decode(x)
        inp[index] = x.squeeze(0).squeeze(-1)
        exact[index] = y.squeeze(0)
        pred[index] = out.squeeze(0)

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
x_test_plot = np.linspace(0, 1, s).astype('float32')
y_test_plot = np.linspace(0, 1, s).astype('float32')
x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)

a_ind = inp[index, :, :]
u_pred = pred[index, :, :, -1]
u_exact = exact[index, :, :, -1]

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

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

for T_index in range(pred.shape[-1]):
    u_pred = pred[index, :, :, T_index]
    plt.contourf(x_test_plot, y_test_plot, u_pred, levels=1, cmap='hsv')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Predicted Value at T_index = {T_index}')
    #plt.show()
    image_path = os.path.join(output_dir, f"frame_{T_index:04d}.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

video_path = "output_video.mp4"
frame_rate = 10  # frames per second

# Get list of saved image paths
image_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])

# Create a video writer object
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

# Write frames to video
for image_file in image_files:
    video.write(cv2.imread(image_file))

video.release()
print(f"Video saved at {video_path}")
