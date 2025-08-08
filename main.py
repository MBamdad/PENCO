import os
import importlib
import torch
import inspect
import numpy as np
import matplotlib
import h5py  # MODIFIED: Added h5py import
import scipy.io

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from training import train_fno, train_fno_time, train_hybrid, train_fno4d
from torch.utils.data import DataLoader, random_split
from utilities import ImportDataset, count_params, LpLoss, ModelEvaluator #, SobolevLoss
from post_processing import plot_loss_trend, plot_combined_results_3d, plot_combined_results, plot_field_trajectory, \
    make_video, save_vtk, plot_xy_plane_subplots
import time
from torch_optimizer import Lamb

################################################################
# Problem Definition
################################################################

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
# problem = 'MBE2D'
#problem = 'MBE3D'
# problem = 'CH2D'
#problem = 'CH3D'

# network_name = 'TNO2d'
# network_name = 'FNO2d'
#network_name = 'FNO3d'
#network_name = 'FNO4d'
network_name = 'TNO3d'

PINN_MODE =  False #True #   True #  False #  True #   True #    True #
#  False #    True # False #  False  # False #

print(f"problem = {problem}")
print(f"network = {network_name}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")
network = getattr(importlib.import_module('networks'), network_name)
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

PDE_WEIGHT = cf.pde_weight
pde_loss_scaler = cf.pde_loss_scaler

if PINN_MODE:
    run_descriptor = f"PINN_w{int(PDE_WEIGHT * 100)}"
    output_subdir = f"plots_Data_Physics_{network_name}"
    model_run_name = f'{network_name}_{problem}_Hybrid_S{cf.s}_T{cf.T_in}to{cf.T_out}_width{cf.width}_modes{cf.modes}_q{cf.width_q}_h{cf.width_h}_grf3d.pt'
else:
    run_descriptor = "DataDriven"
    output_subdir = f"plots_{network_name}"
    model_run_name = f'{network_name}_{problem}_S{cf.s}_T{cf.T_in}to{cf.T_out}_width{cf.width}_modes{cf.modes}_q{cf.width_q}_h{cf.width_h}_grf3d.pt'

model_dir = os.path.join(problem, 'models')
model_name = f'{model_run_name}'
model_path = os.path.join(model_dir, model_name)
plot_dir = os.path.join(problem, output_subdir)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print(f"Model Run Name: {model_run_name}")
print(f"Model Path: {model_path}")
print(f"Plot Directory: {plot_dir}")

start_time = time.time()

################################################################
# load data and data normalization
################################################################
# MODIFIED: Added special handling for SH3D dataset loading
try:
    dataset = ImportDataset(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out)
    if problem == 'SH3D':
        print("SH3D dataset detected - applying special handling")
        # Verify dataset sizes
        sample = dataset[0][0]  # Get first sample
        print(f"SH3D sample shape: {sample.shape}, dtype: {sample.dtype}")
        print(f"SH3D stats - mean: {sample.mean():.4f}, std: {sample.std():.4f}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

train_dataset, test_dataset, _ = random_split(dataset, [cf.nTrain, cf.nTest, len(dataset) - cf.nTrain - cf.nTest])
normalizers = [dataset.normalizer_x, dataset.normalizer_y] if cf.normalized is True else None

train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False)


# ==================== CODE TO INSPECT BATCH SHAPE ====================
print("\n" + "="*50)
print("Inspecting DataLoader Batch Shapes")
print("="*50)
# Get one batch of data from the train_loader
try:
    x_batch, y_batch = next(iter(train_loader))
    # Print the shape of the batch
    # This will be (batch_size, S, S, S, T_in) for input
    # and (batch_size, S, S, S, T_out) for target
    print(f"Shape of an input batch from DataLoader: {x_batch.shape}")
    print(f"Shape of a target batch from DataLoader: {y_batch.shape}")
except StopIteration:
    print("Train loader is empty. Cannot retrieve a batch.")
print("="*50 + "\n")
# =======================================================================

############AA####################################################
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
    model = network(cf.modes, cf.modes, cf.modes, cf.width, cf.width_q, cf.width_h, cf.T_in, cf.T_out, cf.n_layers).to(
        device)
elif network_name == 'FNO4d':
    model = network(
        modes1=cf.modes,
        modes2=cf.modes,
        modes3=cf.modes,
        modes4_internal =1, # cf.modes_t, # MUST BE 1
        width=cf.width,
        width_q=cf.width_q,
        T_in_channels=cf.T_in,
        n_layers=cf.n_layers
    ).to(device)


else:
    raise Exception("network_name is not correct")

print(count_params(model))  # Print model parameters
train_mse_log, train_l2_log, test_l2_log = [], [], []  # Initialize logs

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
myloss = LpLoss(p=2, l1_weight=0.0, size_average=False) # size_average=True # False
# NEW: Instantiate SobolevLoss instead of LpLoss
# You can tune grad_weight. A good starting point is 0.1 or 1.0.
#myloss = SobolevLoss(d=3, p=2, grad_weight=0.1)

# COMPUTE THE DYNAMIC SCALER
# Use the train_loader to get a representative batch


# Train the model
if cf.training:
    print("\n--- Starting Training ---")
    if PINN_MODE:  # Use PINN loop regardless of weight (handles pde_weight=0 case)
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
                f"Running PINN training loop with pde_weight={PDE_WEIGHT:.2f}, using PDE Scaler: {pde_loss_scaler:.2e}")

        model, train_mse_hybrid_log, train_l2_hybrid_log, test_data_log, test_pde_loss_scaled_log, train_data_log, train_pde_scaled_log, test_loss_hybrid_log = (
            train_hybrid(model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
                         optimizer, scheduler, cf.normalized, normalizers, device,
                         PDE_WEIGHT, grid_info, cf.epsilon, problem,
                         pde_loss_scaler=pde_loss_scaler)
        )

        print(f"Saving model and logs to {model_path}")
        torch.save({
            'model': model,
            'train_mse_log': train_mse_hybrid_log,
            'train_l2_log': train_l2_hybrid_log,
            'test_l2_log': test_data_log,
            'test_pde_scaled_log': test_pde_loss_scaled_log,
            'train_data_log': train_data_log,
            'train_pde_scaled_log': train_pde_scaled_log,
            'test_loss_hybrid_log': test_loss_hybrid_log
        }, model_path)

    else:  # Original Data-Driven Mode
        if network_name == 'FNO2d' or network_name == 'FNO3d':
            model, train_l2_log, test_l2_log = (
                train_fno_time(model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
                               optimizer, scheduler, cf.normalized, normalizers, device))
            train_mse_log = []
        elif network_name == 'FNO4d':

            model, train_mse_log, train_l2_log, test_l2_log = (  # Add val logs
                train_fno4d(model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
                          optimizer, scheduler, cf.normalized, normalizers, device))

            # train_fno4d = train_fno
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


'''
# Train the model
if cf.training:
    print("\n--- Starting Training ---")
    if PINN_MODE:  # Use PINN loop regardless of weight (handles pde_weight=0 case)
        grid_info = {
            'Nx': cf.s, 'Ny': cf.s, 'Nz': cf.s,
            'Lx': cf.Lx, 'Ly': cf.Lx, 'Lz': cf.Lx,
            'dt_model': cf.dt_model,
            'T_out': cf.T_out
        }

        # --- NEW: Define the two stages for the training curriculum ---
        epochs_stage1 = 10
        scaler_stage1 = 1e-4

        # Calculate remaining epochs for stage 2
        epochs_stage2 = cf.epochs - epochs_stage1
        scaler_stage2 = 1e-6

        # --- Stage 1 Training ---
        print("\n--- Starting Training Stage 1 ---")
        print(f"Epochs: {epochs_stage1}, PDE Scaler: {scaler_stage1:.2e}, PDE Weight: {PDE_WEIGHT:.2f}")

        # Note: The 'model' object is updated in-place by the function call
        model, train_mse_s1, train_l2_s1, test_data_s1, test_pde_s1, train_data_s1, train_pde_scl_s1, test_loss_s1 = (
            train_hybrid(model, myloss, epochs_stage1, cf.batch_size, train_loader, test_loader,
                         optimizer, scheduler, cf.normalized, normalizers, device,
                         PDE_WEIGHT, grid_info, cf.epsilon, problem,
                         pde_loss_scaler=scaler_stage1)
        )

        # --- Stage 2 Training (if there are remaining epochs) ---
        if epochs_stage2 > 0:
            print("\n--- Starting Training Stage 2 ---")
            print(f"Epochs: {epochs_stage2}, PDE Scaler: {scaler_stage2:.2e}, PDE Weight: {PDE_WEIGHT:.2f}")

            model, train_mse_s2, train_l2_s2, test_data_s2, test_pde_s2, train_data_s2, train_pde_scl_s2, test_loss_s2 = (
                train_hybrid(model, myloss, epochs_stage2, cf.batch_size, train_loader, test_loader,
                             optimizer, scheduler, cf.normalized, normalizers, device,
                             PDE_WEIGHT, grid_info, cf.epsilon, problem,
                             pde_loss_scaler=scaler_stage2)
            )

            # Combine the logs from both stages for plotting and saving
            train_mse_hybrid_log = train_mse_s1 + train_mse_s2
            train_l2_hybrid_log = train_l2_s1 + train_l2_s2
            test_data_log = test_data_s1 + test_data_s2
            test_pde_loss_scaled_log = test_pde_s1 + test_pde_s2
            train_data_log = train_data_s1 + train_data_s2
            train_pde_scaled_log = train_pde_scl_s1 + train_pde_scl_s2
            test_loss_hybrid_log = test_loss_s1 + test_loss_s2
        else:
            # If only stage 1 was run, the final logs are just the stage 1 logs
            train_mse_hybrid_log = train_mse_s1
            train_l2_hybrid_log = train_l2_s1
            test_data_log = test_data_s1
            test_pde_loss_scaled_log = test_pde_s1
            train_data_log = train_data_s1
            train_pde_scaled_log = train_pde_scl_s1
            test_loss_hybrid_log = test_loss_s1

        print(f"\n--- Training Finished. Saving model and logs to {model_path} ---")
        # The torch.save call remains the same, as the log variables have been correctly prepared
        torch.save({
            'model': model,
            'train_mse_log': train_mse_hybrid_log,
            'train_l2_log': train_l2_hybrid_log,
            'test_l2_log': test_data_log, # 'test_l2_log' is used by evaluator, it should contain data loss
            'test_pde_scaled_log': test_pde_loss_scaled_log,
            'train_data_log': train_data_log,
            'train_pde_scaled_log': train_pde_scaled_log,
            'test_loss_hybrid_log': test_loss_hybrid_log
        }, model_path)

    else:  # Original Data-Driven Mode (This part remains unchanged)
        if network_name == 'FNO2d' or network_name == 'FNO3d':
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
'''

end_time = time.time()
Final_time = round(end_time - start_time, 2)
print(f"Total Execution Time: {Final_time} seconds")

# ==================== START: CAPTURE PREDICTION AND EXACT SOLUTION TIMES ====================
print("\n--- Evaluating Model and Measuring Prediction Time ---")
evaluator = ModelEvaluator(model, test_dataset, cf.s, cf.T_in, cf.T_out, device, cf.normalized, normalizers,
                           time_history=(network_name == 'FNO2d'))

# Measure the time it takes to get predictions for the entire test set
prediction_start_time = time.time()
results = evaluator.evaluate(loss_fn=myloss)
prediction_end_time = time.time()

# Calculate the model's prediction time
model_prediction_time = prediction_end_time - prediction_start_time
print(f"Model prediction time for the test set: {model_prediction_time:.4f} seconds")

# IMPORTANT: Placeholder for the exact solution time.
# This value MUST be updated manually with the time it took the numerical
# solver to generate the ground truth data for the test set.
# The value here is just an example.
exact_solution_time = 3600.0  # Placeholder in seconds (e.g., 1 hour)
print(f"Using placeholder for exact solution time: {exact_solution_time:.2f} seconds. PLEASE UPDATE THIS VALUE.")
# ===================== END: CAPTURE PREDICTION AND EXACT SOLUTION TIMES =====================


inp = results['input']
pred = results['prediction']
exact = results['exact']
test_l2_avg = results["average"]

if PINN_MODE:
    losses = [train_l2_hybrid_log, test_loss_hybrid_log, test_data_log, test_pde_loss_scaled_log, train_data_log,
              train_pde_scaled_log]
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

print(f"DEBUG: Shape of u_exact passed to plotting function: {u_exact.shape}")
print(f"DEBUG: Shape of u_pred passed to plotting function: {u_pred.shape}")

plot_range = [[-1.2, 1.2], [-1.2, 1.2], [-0.6, 0.6]]

# =========================================================================================
# ===                START OF MODIFIED PLOTTING PREPARATION BLOCK                      ===
# =========================================================================================

# 1. Get the initial condition (t=0) data for the chosen sample index
a_ind = inp[cf.index]
print(f"Shape of initial condition (t=0) data 'a_ind': {a_ind.shape}")

# 2. Separate desired times into t=0 vs. future predictions
desired_times = cf.time_steps
future_times_to_plot = []
has_initial_condition = (0 in desired_times)
for t in desired_times:
    if t > 0:
        future_times_to_plot.append(t)

# 3. Translate the FUTURE times to array indices
indices_to_plot = []
valid_future_times = []
for t in future_times_to_plot:
    if t <= cf.T_out:
        indices_to_plot.append(t - 1)
        valid_future_times.append(t)
    else:
        print(f"Warning: Time t={t} is out of valid prediction range. Skipping.")
print(f"Plotting for future times: {valid_future_times} --> which correspond to array indices: {indices_to_plot}")

# 4. MOVE ALL DATA TO THE TARGET DEVICE BEFORE OPERATIONS
t0_data_cpu = a_ind
u_exact_cpu = u_exact
u_pred_cpu = u_pred
error_cpu = error

t0_data_gpu = t0_data_cpu.to(device)
u_exact_gpu = u_exact_cpu.to(device)
u_pred_gpu = u_pred_cpu.to(device)
error_gpu = error_cpu.to(device)
indices_tensor_gpu = torch.tensor(indices_to_plot, device=device)

# 5. PREPARE COMBINED TENSORS FOR PLOTTING on the GPU
if has_initial_condition:
    # --- ### FIXED DIMENSION HANDLING ### ---
    # `a_ind` has shape (sx, sy, sz, 1) where the last dim is a channel.
    # `u_exact_gpu` has shape (sx, sy, sz, T) where the last dim is time.
    # They are both 4D, so we can concatenate them directly if the last dimension is treated as the time dimension.
    # We don't need to do any reshaping. `t0_data_gpu` is already correct.
    t0_for_concat = t0_data_gpu

    # Select the future time slices from the GPU tensors
    u_exact_selected = u_exact_gpu.index_select(-1, indices_tensor_gpu)
    u_pred_selected = u_pred_gpu.index_select(-1, indices_tensor_gpu)
    error_selected = error_gpu.index_select(-1, indices_tensor_gpu)

    # Combine t=0 data with the selected future steps
    u_exact_for_plot = torch.cat((t0_for_concat, u_exact_selected), dim=-1)
    u_pred_for_plot = torch.cat((t0_for_concat, u_pred_selected), dim=-1)

    # The error for t=0 is zero by definition
    error_t0 = torch.zeros_like(t0_for_concat)
    error_for_plot = torch.cat((error_t0, error_selected), dim=-1)

    final_indices = list(range(len(desired_times)))
    final_labels = desired_times
else:
    u_exact_for_plot = u_exact_gpu
    u_pred_for_plot = u_pred_gpu
    error_for_plot = error_gpu
    final_indices = indices_to_plot
    final_labels = valid_future_times

print(f"Final data prepared for plotting with shape: {u_exact_for_plot.shape}")
print(f"Final indices for plotting: {final_indices}")
print(f"Final labels for plotting: {final_labels}")


# =========================================================================================
# ===                 END OF MODIFIED PLOTTING PREPARATION BLOCK                       ===
# =========================================================================================

################################################################
# Save Results to MATLAB .mat file
################################################################
print("\n--- Saving Results to .mat File ---")
mat_filename = os.path.join(plot_dir, f'{model_run_name}_results.mat')

# MODIFIED: Enhanced saving logic with fallbacks
def save_results(mat_filename, results_dict):
    try:
        # First try standard save
        scipy.io.savemat(mat_filename, results_dict)
        print(f"Saved with standard format to {mat_filename}")
    except ValueError as e:
        if "Format should be '4' or '5'" in str(e):
            print("Large data detected, trying v7.3 format...")
            try:
                scipy.io.savemat(mat_filename, results_dict, format='v7.3')
                print(f"Saved with v7.3 format to {mat_filename}")
            except Exception as e:
                print(f"v7.3 failed: {e}")
                # Fallback to HDF5
                h5_filename = mat_filename.replace('.mat', '.h5')
                with h5py.File(h5_filename, 'w') as f:
                    for k, v in results_dict.items():
                        f.create_dataset(k, data=v, compression='gzip')
                print(f"Saved as HDF5 to {h5_filename}")
        else:
            raise

if PINN_MODE:
    results_dict = {
        'train_mse_log': np.array(train_mse_hybrid_log, dtype=np.float32),  # MODIFIED: Added dtype
        'train_hybrid_loss': np.array(train_l2_hybrid_log, dtype=np.float32),
        'test_loss_hybrid_log': np.array(test_loss_hybrid_log, dtype=np.float32),
        'train_data_log': np.array(train_data_log, dtype=np.float32),
        'test_data_log': np.array(test_data_log, dtype=np.float32),
        'train_pde_scaled_log': np.array(train_pde_scaled_log, dtype=np.float32),
        'test_pde_loss_scaled_log': np.array(test_pde_loss_scaled_log, dtype=np.float32),
        'test_input': inp.cpu().numpy().astype(np.float32),  # MODIFIED: Added astype
        'test_prediction': pred.cpu().numpy().astype(np.float32),
        'test_exact': exact.cpu().numpy().astype(np.float32),
        'config_pde_weight': np.float32(PDE_WEIGHT),
        'config_pde_loss_scaler': np.float32(pde_loss_scaler),
        'config_epochs': np.int32(cf.epochs),
        'config_lr': np.float32(cf.learning_rate),
        'config_T_in': np.int32(cf.T_in),
        'config_T_out': np.int32(cf.T_out),
        'config_s': np.int32(cf.s),
        'config_Lx': np.float32(cf.Lx),
        'final_exec_time_s': np.float32(Final_time),
        'model_prediction_time': np.float32(model_prediction_time),  # ADDED
        'exact_solution_time': np.float32(exact_solution_time),      # ADDED
    }
else:
    results_dict = {
        'train_mse_log': np.array(train_mse_log, dtype=np.float32),
        'train_l2_log': np.array(train_l2_log, dtype=np.float32),
        'test_l2_log': np.array(test_l2_log, dtype=np.float32),
        'test_input': inp.cpu().numpy().astype(np.float32),
        'test_prediction': pred.cpu().numpy().astype(np.float32),
        'test_exact': exact.cpu().numpy().astype(np.float32),
        'config_epochs': np.int32(cf.epochs),
        'config_lr': np.float32(cf.learning_rate),
        'config_T_in': np.int32(cf.T_in),
        'config_T_out': np.int32(cf.T_out),
        'config_s': np.int32(cf.s),
        'config_Lx': np.float32(cf.Lx),
        'final_exec_time_s': np.float32(Final_time),
        'model_prediction_time': np.float32(model_prediction_time),  # ADDED
        'exact_solution_time': np.float32(exact_solution_time),      # ADDED
    }

# MODIFIED: Use the new save function
save_results(mat_filename, results_dict)

# Plot XY-plane for the "Exact" solution trajectory (includes t=0)

'''
plot_xy_plane_subplots(domain=cf.domain,
                       field=u_exact_for_plot,
                       field_name='Exact Solution',
                       time_steps=final_indices,
                       desired_times=final_labels,
                       plot_range=plot_range[0],
                       problem=problem,
                       network_name=network_name)

# Plot XY-plane for the "Predicted" solution trajectory (includes t=0 from input)
plot_xy_plane_subplots(domain=cf.domain,
                       field=u_pred_for_plot,
                       field_name='Predicted Solution',
                       time_steps=final_indices,
                       desired_times=final_labels,
                       plot_range=plot_range[1],
                       problem=problem,
                       network_name=network_name)

# Plot XY-plane for the "Error" (error is 0 at t=0)
plot_xy_plane_subplots(domain=cf.domain,
                       field=error_for_plot,
                       field_name='Error',
                       time_steps=final_indices,
                       desired_times=final_labels,
                       plot_range=plot_range[2],
                       problem=problem,
                       network_name=network_name)
'''

# Calculate L2 norm on original full prediction
l2_norm_error = torch.norm(u_pred - u_exact, p=2)
l2_norm_exact = torch.norm(u_exact, p=2)
epsilon = 1e-8
if l2_norm_exact.item() > epsilon:
    relative_l2_error = l2_norm_error / l2_norm_exact
else:
    relative_l2_error = torch.tensor(0.0) if l2_norm_error.item() < epsilon else torch.tensor(float('inf'))

print(f"L2 norm of error: {l2_norm_error.item()}")
print(f"L2 norm of exact solution: {l2_norm_exact.item()}")
print(f"Relative L2 norm error: {relative_l2_error.item()}")
relative_l2_error_percentage = (relative_l2_error * 100)
print(f"Relative L2 norm error (percentage): {relative_l2_error_percentage.item()}%")

# Call the combined results plots with the prepared data
plot_combined_results(
    domain=cf.domain,
    u_exact=u_exact_for_plot,
    u_pred=u_pred_for_plot,
    error=error_for_plot,
    plot_ranges=plot_range,
    problem=problem,
    network_name=network_name,
    plot_dir=plot_dir,
    pde_weight=PDE_WEIGHT,
    time_steps_indices=final_indices,
    desired_times=final_labels
)

plot_combined_results_3d(
    domain=cf.domain,
    u_exact=u_exact_for_plot,
    u_pred=u_pred_for_plot,
    error=error_for_plot,
    plot_ranges=plot_range,
    problem=problem,
    network_name=network_name,
    plot_dir=plot_dir,
    pde_weight=PDE_WEIGHT,
    time_steps_indices=final_indices,
    desired_times=final_labels
)

print("\n--- Script Finished ---")