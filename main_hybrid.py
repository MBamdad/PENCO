# main_hybrid.py

import os
import importlib
import torch
import numpy as np
import scipy.io
import time
from torch.utils.data import DataLoader, random_split

# --- Import our custom modules ---
from trainer_hybrid import train_fno, train_hybrid_fno4d
from utilities import ImportDataset, count_params, LpLoss, ModelEvaluator
from post_processing import plot_loss_trend, plot_combined_results_3d

# --- Problem & Model Definition ---
problem = 'AC3D'
network_name = 'FNO4d'
PINN_MODE = True  # Set to True for hybrid, False for pure data-driven

print(f"Problem: {problem}")
print(f"Network: {network_name}")
print(f"Mode: {'Hybrid (Data+Physics)' if PINN_MODE else 'Data-Driven'}")

# --- Load Configuration ---
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")
network = getattr(importlib.import_module('networks_hybrid'), network_name)

# --- Setup Environment ---
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# --- File Naming and Directories ---
if PINN_MODE:
    run_descriptor = f"PINN_w{int(cf.pde_weight * 100)}"
    output_subdir = f"plots_Data_Physics_{network_name}"
    model_run_name = f'{network_name}_{problem}_Hybrid_w{int(cf.pde_weight*100)}_n{cf.nTrain}.pt'
else:
    run_descriptor = "DataDriven"
    output_subdir = f"plots_{network_name}"
    model_run_name = f'{network_name}_{problem}_Data_n{cf.nTrain}.pt'

model_dir = os.path.join(problem, 'models')
plot_dir = os.path.join(problem, output_subdir)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_run_name)

print(f"Model Run Name: {model_run_name}")
print(f"Saving plots to: {plot_dir}")
print(f"Model path: {model_path}")

start_time = time.time()

# --- Load Data ---
dataset = ImportDataset(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out)
train_dataset, test_dataset, _ = random_split(dataset, [cf.nTrain, cf.nTest, len(dataset) - cf.nTrain - cf.nTest])

train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False)

normalizers = [dataset.normalizer_x, dataset.normalizer_y] if cf.normalized else None
if cf.normalized and normalizers is not None:
    print("Moving normalizers to device...")
    if normalizers[0] is not None: normalizers[0].to(device)
    if normalizers[1] is not None: normalizers[1].to(device)

# --- Initialize Model ---
model = network(
    modes1=cf.modes,
    modes2=cf.modes,
    modes3=cf.modes,
    modes4_internal=1, # Not used in this FNO4d version
    width=cf.width,
    width_q=cf.width_q,
    T_in_channels=cf.T_in,
    n_layers=cf.n_layers
).to(device)

print(f"Total Trainable Parameters: {count_params(model):,}")

# --- Optimizer, Scheduler, Loss ---
optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.epochs)
data_loss_fn = LpLoss(p=2, size_average=False)

if os.path.exists(model_path) and cf.load_model:
    print(f"Loading pre-trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Initializing a new model.")

# --- Training ---
if cf.training:
    print("\n--- Starting Training ---")
    if PINN_MODE:
        # This dictionary passes all necessary physical parameters to the trainer
        grid_info = {
            'Nx': cf.s, 'Ny': cf.s, 'Nz': cf.s,
            'Lx': cf.Lx, 'Ly': cf.Ly, 'Lz': cf.Lz,
            'dt_model': cf.dt_model,
            'epsilon': cf.epsilon,
            'Cahn': cf.epsilon**2
        }
        print(f"Running Hybrid training with PDE weight = {cf.pde_weight:.3f}")
        model, logs = train_hybrid_fno4d(
            model, data_loss_fn, cf.epochs, train_loader, test_loader,
            optimizer, scheduler, cf.normalized, normalizers, device,
            pde_weight=cf.pde_weight,
            grid_info=grid_info
        )
    else: # Data-Driven Mode
        print("Running Data-Driven training...")
        model, logs = train_fno(
            model, data_loss_fn, cf.epochs, train_loader, test_loader,
            optimizer, scheduler, cf.normalized, normalizers, device
        )

    print(f"Saving trained model and logs to {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'logs': logs,
    }, model_path)

# --- Post-Training Analysis ---
end_time = time.time()
print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

# --- Post-Training Analysis ---
end_time = time.time()
Final_time = round(end_time - start_time, 2)
print(f"\nTotal Execution Time: {Final_time:.2f} seconds")

evaluator = ModelEvaluator(model, test_dataset, cf.s, cf.T_in, cf.T_out, device, cf.normalized, normalizers)
results = evaluator.evaluate(loss_fn=data_loss_fn)
test_l2_avg = results["average"]
print(f"Final Average Test L2 Error: {test_l2_avg:.4e}")

# =======================================================================
# === START OF CORRECTION FOR PLOTTING ===
# =======================================================================
print("\n--- Generating Loss Plots ---")
# Prepare the data for the plotting function in the expected format
if PINN_MODE:
    losses = [
        logs['total_loss'],
        logs['data_loss'],
        logs['pde_loss'],
        logs['test_l2']
    ]
    labels = [
        'Total Train Loss',
        'Train Data Loss',
        'Train PDE Loss',
        'Test L2 Error'
    ]
else: # Data-Driven Mode
    losses = [
        logs['data_loss'], # This is the training L2 loss
        logs['test_l2']
    ]
    labels = [
        'Train L2 Error',
        'Test L2 Error'
    ]

# Now, call the plotting function with all the required arguments
# Note: We assume plot_loss_trend has a signature like:
# plot_loss_trend(losses, labels, problem, network_name, Final_time, test_l2_avg, plot_dir, pde_weight)
# which matches your original code's usage.
plot_loss_trend(
    losses,
    labels,
    problem,
    network_name,
    Final_time,
    test_l2_avg,
    plot_dir,
    cf.pde_weight if PINN_MODE else 0.0
)
print("Loss plots saved successfully.")
# =======================================================================
# === END OF CORRECTION FOR PLOTTING ===
# =======================================================================


# --- Visualize a Sample ---
print("\n--- Visualizing a Sample Trajectory ---")
sample_idx = cf.index
inp = results['input'][sample_idx]
pred = results['prediction'][sample_idx]
exact = results['exact'][sample_idx]
error = pred - exact

# Combine input (t=0) with the predicted/exact future steps for a full trajectory
u_exact_full = torch.cat((inp, exact), dim=-1)
u_pred_full = torch.cat((inp, pred), dim=-1)
error_full = torch.cat((torch.zeros_like(inp), error), dim=-1) # Error at t=0 is zero

# Select time steps to plot from the full trajectory
# Example: plot time steps 0, 50, 99 from a 101-step trajectory (T_in=1, T_out=100)
time_steps_to_plot = cf.time_steps

plot_combined_results_3d(
     domain=cf.domain,
     u_exact=u_exact_full,
     u_pred=u_pred_full,
     error=error_full,
     plot_ranges=[[-1.2, 1.2], [-1.2, 1.2], [-0.6, 0.6]],
     problem=problem,
     network_name=network_name,
     plot_dir=plot_dir,
     pde_weight=cf.pde_weight if PINN_MODE else 0.0,
     time_steps_indices=time_steps_to_plot, # Pass the indices directly
     desired_times=time_steps_to_plot
)
print("Visualization plots saved successfully.")
print("\n--- Script Finished ---")