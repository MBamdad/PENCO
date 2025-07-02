import os
import importlib
import torch
import inspect
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from training import train_fno, train_fno_time, train_hybrid
from torch.utils.data import DataLoader, random_split
from utilities import ImportDataset, count_params, LpLoss, ModelEvaluator
from post_processing import plot_loss_trend, plot_combined_results_3d, plot_combined_results, plot_field_trajectory, make_video, save_vtk, plot_xy_plane_subplots
import time  # Import the time module at the beginning of the script
from torch_optimizer import Lamb
import scipy.io

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

PINN_MODE = False # True # False #

print(f"problem = {problem}")
print(f"network = {network_name}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cf = importlib.import_module(f"configs.config_{problem}_{network_name}") # configuration file
# above line means: import configs.config_PFC3D_TNO3d as cf
network = getattr(importlib.import_module('networks'), network_name) # from networks import TNO3d
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)
#device = torch.device(cf.gpu_number if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# --- Define Output Directory ---


PDE_WEIGHT = cf.pde_weight
PDE_LOSS_SCALER = cf.pde_loss_scaler

if PINN_MODE:
    run_descriptor = f"PINN_w{int(PDE_WEIGHT * 100)}"
    output_subdir = f"plots_Data_Physics_{network_name}"  # Specific PINN output
else:
    run_descriptor = "DataDriven"
    output_subdir = f"plots_{network_name}"  # Original data-driven output

#model_run_name = f'{network_name}_{problem}_S{cf.s}_T{cf.T_in}to{cf.T_out}_w{cf.width}_m{cf.modes}_q{cf.width_q}_h{cf.width_h}_{run_descriptor}'
model_run_name = f'{network_name}_{problem}_S{cf.s}_T{cf.T_in}to{cf.T_out}_width{cf.width}_modes{cf.modes}_q{cf.width_q}_h{cf.width_h}.pt'
model_dir = os.path.join(problem, 'models') # models_smpooth
model_name = f'{model_run_name}'
model_path = os.path.join(model_dir, model_name)

plot_dir = os.path.join(problem, output_subdir)  # Use specific plot dir
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)  # Make sure plot directory exists

print(f"Model Run Name: {model_run_name}")
print(f"Model Path: {model_path}")
print(f"Plot Directory: {plot_dir}")

# width_q = 32
start_time = time.time()

################################################################
# load data and data normalization
################################################################
#model_dir = problem + '/models'

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
    model.load_state_dict(checkpoint['model_state_dict']) # Use this line if you only save state_dict
    #model = checkpoint['model'] # Use this line if you save the entire model object
    train_mse_log = checkpoint.get('train_mse_log', [])
    train_l2_log = checkpoint.get('train_l2_log', [])
    test_l2_log = checkpoint.get('test_l2_log', [])
else:
    print("No pre-trained model loaded. Initializing a new model.")

# Define optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.iterations)
myloss = LpLoss(size_average=False)

###

# Train the model
if cf.training:
    print("\n--- Starting Training ---")
    if PINN_MODE:
        grid_info = {
            'Nx': cf.s, 'Ny': cf.s, 'Nz': cf.s,
            'Lx': cf.Lx, 'Ly': cf.Lx, 'Lz': cf.Lx,
            'dt_model': cf.dt_model,
            'T_out': cf.T_out
        }

        if PDE_WEIGHT == 0.0:
            print("Running PINN training loop with pde_weight=0 (Data-Driven only loss).")
        else:
            print(
                f"Running PINN training loop with pde_weight={PDE_WEIGHT:.2f}, using PDE Scaler: {PDE_LOSS_SCALER:.2e}")

        model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log = (
            train_hybrid(model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
                         optimizer, scheduler, cf.normalized, normalizers, device,
                         PDE_WEIGHT, grid_info, cf.epsilon, problem,
                         pde_loss_scaler=PDE_LOSS_SCALER)
        )

        print(f"Saving model and logs to {model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_mse_log': train_mse_hybrid_log,
            'train_l2_log': train_l2_hybrid_log,
            'test_l2_log': test_data_log,
            'test_pde_scaled_log': test_pde_loss_scaled_log,
            'train_data_log': train_data_log,
            'train_pde_scaled_log': train_pde_scaled_log,
            'test_loss_hybrid_log': test_loss_hybrid_log
        }, model_path)

    else:
        if network_name in ['FNO2d', 'FNO3d']:
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
            'model_state_dict': model.state_dict(),
            'train_mse_log': train_mse_log,
            'train_l2_log': train_l2_log,
            'test_l2_log': test_l2_log
        }, model_path)


end_time = time.time()
Final_time = round(end_time - start_time, 2)
print(f"Total Execution Time: {Final_time} seconds")

evaluator = ModelEvaluator(model, test_dataset, cf.s, cf.T_in, cf.T_out, device, cf.normalized, normalizers,
                           time_history=(network_name in ['FNO2d', 'FNO3d']))

results = evaluator.evaluate(loss_fn=myloss)
inp = results['input']
pred = results['prediction']
exact = results['exact']
test_l2_avg = results["average"]

if PINN_MODE:
    losses = [train_l2_hybrid_log, test_loss_hybrid_log, test_data_log, test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log]
    labels = ['Train L2 Hyb', 'Test L2 Hyb', 'Test L2 data', 'Test pde_scaled', 'Train data', 'Train pde']
    plot_loss_trend(losses, labels, problem, network_name, Final_time, test_l2_avg, plot_dir, PDE_WEIGHT)
else:
    losses = [train_l2_log, test_l2_log]
    labels = ['Train L2', 'Test L2']
    plot_loss_trend(losses, labels, problem, network_name, Final_time, test_l2_avg, plot_dir, PDE_WEIGHT)


################################################################
# post-processing
################################################################
a_ind = inp[cf.index]
u_pred = pred[cf.index]
u_exact = exact[cf.index]
error = u_pred - u_exact

plot_range = [[-1.2, 1.2], [-1.2, 1.2], [-0.6, 0.6]]
print(f"Field shape: {u_exact.shape}")

selected_time_steps = [0, 2, 4, 6, 8, 9]

# Plot exact solution
plot_xy_plane_subplots(domain=cf.domain,
                      field=u_exact,
                      field_name='Exact Solution',
                      time_steps=selected_time_steps,
                      plot_range=plot_range[0],
                      problem=problem,
                      network_name=network_name)

# Plot predicted solution
plot_xy_plane_subplots(domain=cf.domain,
                      field=u_pred,
                      field_name='Predicted Solution',
                      time_steps=selected_time_steps,
                      plot_range=plot_range[1],
                      problem=problem,
                      network_name=network_name)

# Plot error
plot_xy_plane_subplots(domain=cf.domain,
                      field=error,
                      field_name='Error',
                      time_steps=selected_time_steps,
                      plot_range=plot_range[2],
                      problem=problem,
                      network_name=network_name)


# ==============================================================================
# NEW SECTION: PREDICT AND VISUALIZE A SINGLE TRAJECTORY EVOLUTION
# ==============================================================================
print("\n--- Generating Visualization for a Single Predicted Trajectory ---")

# Ensure dt_simulation is defined in the config file.
if not hasattr(cf, 'dt_simulation'):
    raise AttributeError("Configuration file is missing 'dt_simulation'. Please add it (e.g., dt_simulation = 0.00002).")

predicted_trajectory = u_pred # Shape: (s, s, s, T_out)

# 1. Choose 4 suitable time frames for visualization from the predicted steps.
num_time_frames_to_plot = 4
total_predicted_steps = predicted_trajectory.shape[-1]  # This is T_out

if total_predicted_steps < 1:
    print("No time steps to plot in the predicted trajectory.")
else:
    # Select 4 evenly spaced indices from the available time steps.
    time_indices_to_plot = np.linspace(0, total_predicted_steps - 1, num_time_frames_to_plot, dtype=int).tolist()

    print(f"Selected time indices for plotting: {time_indices_to_plot}")

    # 2. Create the subplot (1 row, 4 columns) and save it.
    fig, axes = plt.subplots(1, num_time_frames_to_plot, figsize=(5 * num_time_frames_to_plot, 5), squeeze=False)
    axes = axes.flatten()

    vmin = predicted_trajectory.cpu().numpy().min()
    vmax = predicted_trajectory.cpu().numpy().max()
    s = cf.s
    slice_index = s // 2  # Middle slice in the Z-direction

    for i, t_idx in enumerate(time_indices_to_plot):
        # 3. Calculate the correct physical time for the label.
        # The prediction starts after T_in steps.
        physical_time = (cf.T_in + t_idx) * cf.dt_simulation

        slice_2d = predicted_trajectory[:, :, slice_index, t_idx].cpu().numpy()

        ax = axes[i]
        im = ax.imshow(slice_2d, cmap='viridis', vmin=vmin, vmax=vmax,
                       extent=[0, cf.Lx, 0, cf.Ly], origin='lower', interpolation='bicubic')
        ax.set_title(f'Predicted t={physical_time:.2e}') # Use scientific notation for time
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')

    fig.colorbar(im, ax=axes.tolist(), orientation='vertical', pad=0.02)
    z_coord = cf.Lx / s * (slice_index - s / 2)
    fig.suptitle(f'Predicted Evolution (Z-slice at z={z_coord:.2f})', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    subplot_filename = os.path.join(plot_dir, f'{model_run_name}_single_trajectory_subplot.png')
    plt.savefig(subplot_filename, dpi=300, bbox_inches='tight')
    print(f"Trajectory subplot saved to {subplot_filename}")
    plt.close(fig)

# END OF NEW SECTION
# ==============================================================================

# The p=2 explicitly specifies the L2 norm.
l2_norm_error = torch.norm(u_pred - u_exact, p=2)

# Calculate the L2 norm of the exact solution
l2_norm_exact = torch.norm(u_exact, p=2)

# Calculate the relative L2 norm error
epsilon = 1e-8
if l2_norm_exact.item() > epsilon:
    relative_l2_error = l2_norm_error / l2_norm_exact
else:
    if l2_norm_error.item() < epsilon:
        relative_l2_error = torch.tensor(0.0, device=u_pred.device, dtype=u_pred.dtype)
    else:
        print(f"Warning: L2 norm of exact solution is {l2_norm_exact.item()}, which is close to zero. L2 norm of error is {l2_norm_error.item()}.")
        relative_l2_error = torch.tensor(float('inf'), device=u_pred.device, dtype=u_pred.dtype)

print(f"L2 norm of error: {l2_norm_error.item()}")
print(f"L2 norm of exact solution: {l2_norm_exact.item()}")
print(f"Relative L2 norm error: {relative_l2_error.item()}")
relative_l2_error_percentage = (relative_l2_error * 100)
print(f"Relative L2 norm error (percentage): {relative_l2_error_percentage.item()}%")

###
plot_combined_results(
    domain=cf.domain,
    u_exact=u_exact,
    u_pred=u_pred,
    error=error,
    plot_ranges=[
        [-1.2, 1.2],
        [-1.2, 1.2],
        [-1.2, 1.2]
    ],
    problem=problem,
    network_name=network_name,
    plot_dir = plot_dir,
    pde_weight = PDE_WEIGHT
)

plot_combined_results_3d(
    domain=cf.domain,
    u_exact=u_exact,
    u_pred=u_pred,
    error=error,
    plot_ranges=[
        [-1.2, 1.2],
        [-1.2, 1.2],
        [-1.2, 1.2]
    ],
    problem=problem,
    network_name=network_name,
    plot_dir = plot_dir,
    pde_weight = PDE_WEIGHT
)

################################################################
# Save Results to MATLAB .mat file
################################################################
print("\n--- Saving Results to .mat File ---")

mat_filename = os.path.join(plot_dir, f'{model_run_name}_results.mat')

if PINN_MODE:
    try:
        results_dict = {
            'train_mse_log': train_mse_hybrid_log,
            'train_hybrid_loss': np.array(train_l2_hybrid_log),
            'test_loss_hybrid_log': np.array(test_loss_hybrid_log),
            'train_data_log': np.array(train_data_log),
            'test_data_log': np.array(test_data_log),
            'train_pde_scaled_log': np.array(train_pde_scaled_log),
            'test_pde_loss_scaled_log': np.array(test_pde_loss_scaled_log),
            'test_input': inp.cpu().numpy(),
            'test_prediction': pred.cpu().numpy(),
            'test_exact': exact.cpu().numpy(),
            'config_pde_weight': PDE_WEIGHT if PINN_MODE else 0.0,
            'config_pde_loss_scaler': PDE_LOSS_SCALER if PINN_MODE else 0.0,
            'config_epochs': cf.epochs,
            'config_lr': cf.learning_rate,
            'config_T_in': cf.T_in,
            'config_T_out': cf.T_out,
            'config_s': cf.s,
            'config_Lx': cf.Lx,
            'final_exec_time_s': Final_time,
        }
        scipy.io.savemat(mat_filename, results_dict)
        print(f"Results saved successfully to: {mat_filename}")
    except Exception as e:
        print(f"Error saving results to .mat file: {e}")
else:
    try:
        results_dict = {
            'train_mse_log': np.array(train_mse_log),
            'train_l2_log': np.array(train_l2_log),
            'test_l2_log': np.array(test_l2_log),
            'test_input': inp.cpu().numpy(),
            'test_prediction': pred.cpu().numpy(),
            'test_exact': exact.cpu().numpy(),
            'config_epochs': cf.epochs,
            'config_lr': cf.learning_rate,
            'config_T_in': cf.T_in,
            'config_T_out': cf.T_out,
            'config_s': cf.s,
            'config_Lx': cf.Lx,
            'final_exec_time_s': Final_time,
        }
        scipy.io.savemat(mat_filename, results_dict)
        print(f"Results saved successfully to: {mat_filename}")
    except Exception as e:
        print(f"Error saving results to .mat file: {e}")

print("\n--- Script Finished ---")