import os
import importlib
import torch
import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from training import train_fno, train_fno_time
from torch.utils.data import DataLoader, random_split
from utilities import ImportDataset, count_params, LpLoss, ModelEvaluator
from post_processing import plot_loss_trend, plot_field_trajectory, make_video, save_vtk
import time  # Import the time module at the beginning of the script
from torch_optimizer import Lamb
################################################################
# Problem Definition
################################################################
# problem = 'AC2D'
#problem = 'AC3D'
# problem = 'CH2DNL'
# problem = 'SH2D'
problem = 'SH3D'
# problem = 'PFC2D'
#problem = 'PFC3D'
#problem = 'MBE2D'
#problem = 'MBE3D'
# problem = 'CH2D'
#problem = 'CH3D'

#network_name = 'TNO2d'
# network_name = 'FNO2d'
#network_name = 'FNO3d'
network_name = 'TNO3d'

print(f"problem = {problem}")
print(f"network = {network_name}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cf = importlib.import_module(f"configs.config_{problem}_{network_name}") # configuration file
# above line means: import configs.config_PFC3D_TNO3d as cf
network = getattr(importlib.import_module('networks'), network_name) # from networks import TNO3d
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)
#device = torch.device(cf.gpu_number if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# width_q = 32
start_time = time.time()

################################################################
# load data and data normalization
################################################################
model_dir = problem + '/models'
model_name = f'{network_name}_{problem}_S{cf.s}_T{cf.T_in}to{cf.T_out}_width{cf.width}_modes{cf.modes}_q{cf.width_q}_h{cf.width_h}.pt'
print(f"model = {model_name}")
print(f"number of epoch = {cf.epochs}")
print(f"batch size = {cf.batch_size}")
print(f"nTrain = {cf.nTrain}")
print(f"nTest = {cf.nTest}")
print(f"learning_rate = {cf.learning_rate}")
print(f"n_layers = {cf.n_layers}")
print(f"width_q = {cf.width_q}")
print(f"width_h = {cf.width_h}")

model_path = os.path.join(model_dir, model_name)
os.makedirs(model_dir, exist_ok=True)
# dataset creates an instance of the ImportDataset class, initializing it with parameters from cf
dataset = ImportDataset(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out)

train_dataset, test_dataset, _ = random_split(dataset, [cf.nTrain, cf.nTest, len(dataset) - cf.nTrain - cf.nTest])
# to see train_dataset values: print("train_dataset subset:", [train_dataset[i] for i in range(len(train_dataset))])
normalizers = [dataset.normalizer_x, dataset.normalizer_y] if cf.normalized is True else None

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False)
################################################################
# training and evaluation
################################################################
sig = inspect.signature(network.__init__)
required_args = [param.name for param in sig.parameters.values()
                 if param.default == inspect.Parameter.empty and param.name != "self"]
if network_name == 'FNO2d':
    model = network(cf.modes, cf.modes, cf.width, cf.width_q, cf.T_in, cf.T_out, cf.n_layers).to(device)
elif network_name == 'TNO2d':
    model = network(cf.modes, cf.modes, cf.width, cf.width_q, cf.width_h, cf.T_in, cf.T_out, cf.n_layers).to(device)
elif network_name == 'FNO3d':
    model = network(cf.modes, cf.modes, cf.modes, cf.width, cf.width_q, cf.T_in, cf.T_out, cf.n_layers).to(device)
elif network_name == 'TNO3d':
    model = network(cf.modes, cf.modes, cf.modes, cf.width, cf.width_q, cf.width_h, cf.T_in, cf.T_out, cf.n_layers).to(device)
else:
    raise Exception("network_name is not correct")

print(count_params(model))      # Print model parameters
train_mse_log, train_l2_log, test_l2_log = [], [], []       # Initialize logs

# Load the entire model and logs
if os.path.exists(model_path) and cf.load_model:
    print(f"Loading pre-trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    train_mse_log = checkpoint.get('train_mse_log', [])
    train_l2_log = checkpoint.get('train_l2_log', [])
    test_l2_log = checkpoint.get('test_l2_log', [])
else:
    print("No pre-trained model loaded. Initializing a new model.")

# Define optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.iterations)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
#optimizer = Lamb(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

myloss = LpLoss(size_average=False)

# Train the model
if cf.training:
    if network_name == 'FNO2d'  or network_name == 'FNO3d':
        model, train_l2_log, test_l2_log = (
            train_fno_time(model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
                           optimizer, scheduler, cf.normalized, normalizers, device))
        train_mse_log = []
    else:
        model, train_mse_log, train_l2_log, test_l2_log = (
            train_fno(model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
                      optimizer, scheduler, cf.normalized, normalizers, device))
    print(f"Saving model and logs to {model_path}")
    torch.save({
        'model': model,
        'train_mse_log': train_mse_log,
        'train_l2_log': train_l2_log,
        'test_l2_log': test_l2_log
    }, model_path)

# losses = [train_mse_log, train_l2_log, test_l2_log]
# labels = ['Train MSE', 'Train L2', 'Test L2']

end_time = time.time()
Final_time = round(end_time - start_time, 2)
# Print the total execution time
print(f"Total Execution Time: {Final_time} seconds")

evaluator = ModelEvaluator(model, test_dataset, cf.s, cf.T_in, cf.T_out, device, cf.normalized, normalizers,
                           time_history=(network_name == 'FNO2d'))

results = evaluator.evaluate(loss_fn=myloss)
inp = results['input']
pred = results['prediction']
exact = results['exact']
test_l2_avg = results["average"]
# Record the end time


losses = [train_l2_log, test_l2_log]
labels = ['Train L2', 'Test L2']
plot_loss_trend(losses, labels, problem, network_name, Final_time, test_l2_avg)
################################################################
# post-processing
################################################################
a_ind = inp[cf.index]
# plot_field_trajectory(cf.domain, [a_ind], ['Initial Value'], [0], [[-0.2, 0.2]], problem)
u_pred = pred[cf.index]
u_exact = exact[cf.index]

# error = torch.abs(u_pred-u_exact)
error = u_pred-u_exact

# u_pred_e = torch.where(u_pred < -0.0, -1, torch.where(u_pred > 0.0, 1, 0))
# u_exact_e = torch.where(u_exact < -0.0, -1, torch.where(u_exact > 0.0, 1, 0))
# error_e = torch.abs(u_pred_e-u_exact_e)
# error = torch.where(error_e < 0.01, 0, torch.abs(u_pred-u_exact))

# error = torch.abs(u_pred-u_exact)

# Save as VTK files
# vtk_dir = os.path.join(problem, 'vtk_outputs')
# os.makedirs(vtk_dir, exist_ok=True)
# save_vtk(os.path.join(vtk_dir, 'u_pred.vti'), u_pred.cpu().numpy(), u_pred.cpu().numpy().shape)
# save_vtk(os.path.join(vtk_dir, 'u_exact.vti'), u_exact.cpu().numpy(), u_exact.cpu().numpy().shape)
# save_vtk(os.path.join(vtk_dir, 'error.vti'), error.cpu().numpy(), error.cpu().numpy().shape)

fields = [u_exact, u_pred, error]
field_names = ['Exact Value', 'Predicted Value', 'Error']
# plot_range = [[-0.75, 0.75], [-0.75, 0.75], [-0.5, 0.5]]
plot_range = [[-1.2, 1.2], [-1.2, 1.2], [-0.6, 0.6]]
# plot_range = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]]
plot_field_trajectory(cf.domain, fields, field_names, cf.time_steps, plot_range, problem, network_name, plot_show=True,
                          interpolation=True)

# make_video(u_pred, cf.domain, "predicted", plot_range, problem)
# make_video(u_exact, cf.domain, "exact", plot_range, problem)

