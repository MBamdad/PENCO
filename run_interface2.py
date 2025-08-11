import torch
import numpy as np
import importlib
import os
import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' if you are running in a non-GUI environment
import matplotlib.pyplot as plt
from scipy.io import savemat

# Added imports for data handling, splitting, and evaluation
from utilities import ImportDataset, LpLoss
from torch.utils.data import DataLoader, random_split

# ============================================================================
# 1. CHOOSE CASE STUDY AND MODEL APPROACH
# ============================================================================
case_study = 'CH3D'    # Options: 'SH3D', 'AC3D', 'CH3D', 'MBE3D', 'PFC3D'
approach = 'FNO4d'  # Options: 'MHNO', 'PI_MHNO', 'FNO4d'

# Validate the approach choice
if approach not in ['MHNO', 'PI_MHNO', 'FNO4d']:
    raise ValueError(f"Unknown approach: '{approach}'. Please choose 'standard' or 'hybrid'.")

print(f"Running Case Study: {case_study.upper()}")
print(f"Selected Model Approach: {approach.upper()}")

# ============================================================================
# 2. MODEL AND DATASET LOADING
# ============================================================================
# Load model
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model paths for each case study
model_paths = {
    'AC3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d' : '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/FNO4d_AC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'CH3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d' : '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/FNO4d_CH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'SH3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d' : '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/FNO4d_SH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'MBE3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d' : '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/FNO4d_MBE3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'PFC3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d' : '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/FNO4d_PFC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    }
}

# Automatically select the model path
model_path = model_paths[case_study][approach]
print(f"Loading model: {model_path}")

try:
    from networks import TNO3d
    torch.serialization.add_safe_globals([set, TNO3d])
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
except Exception as e:
    print(f"Secure loading failed: {e}\nFalling back to weights_only=False")
    from networks import TNO3d
    torch.serialization.add_safe_globals([set, TNO3d])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

model = checkpoint['model']
model.eval()

# Load configuration and full dataset
problem = case_study
network_name = 'TNO3d'
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")

# Set seeds for reproducible train/test split
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)

# Load the full dataset
dataset = ImportDataset(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out)

# Send normalizers to device
dataset.normalizer_x.mean = dataset.normalizer_x.mean.to(device)
dataset.normalizer_x.std = dataset.normalizer_x.std.to(device)
dataset.normalizer_y.mean = dataset.normalizer_y.mean.to(device)
dataset.normalizer_y.std = dataset.normalizer_y.std.to(device)

# Split dataset to get the same test set used during training
train_size = cf.nTrain
test_size = cf.nTest
if train_size + test_size > len(dataset):
    print(f"Warning: nTrain ({train_size}) + nTest ({test_size}) > total dataset size ({len(dataset)}). Adjusting sizes.")
    test_size = len(dataset) - train_size

train_dataset, test_dataset, _ = random_split(
    dataset, [train_size, test_size, len(dataset) - train_size - test_size]
)

# Create a DataLoader for the test set
test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False)
print(f"Loaded and split dataset. Using {len(test_dataset)} samples for testing.")


# ============================================================================
# 3. EVALUATION ON TEST DATA AND ERROR CALCULATION
# ============================================================================
# Define the absolute time frames for evaluation
start_frame, end_frame = 20, 80
time_frames_to_eval = np.arange(start_frame, end_frame + 1, 5)

# Convert absolute time frames to 0-based indices in the prediction tensor
# The model predicts from T_in+1. Index `i` in the output corresponds to time `T_in + 1 + i`.
# Therefore, the index for a given time `t` is `i = t - T_in - 1`.
output_indices = time_frames_to_eval - cf.T_in - 1

# Validate that the requested time frames are within the model's prediction range
if np.any(output_indices < 0) or np.any(output_indices >= cf.T_out):
    raise ValueError(f"Requested time frames [{start_frame}, {end_frame}] are out of the model's "
                     f"prediction range [1, {cf.T_out}] for T_in={cf.T_in}.")

# Initialize L2 loss calculator for relative error
l2_loss_criterion = LpLoss(size_average=False)
avg_errors_per_frame = []

print("\nStarting evaluation on the test set...")
model.eval()
with torch.no_grad():
    # Loop over each time frame we need to evaluate
    for i, time_idx in enumerate(output_indices):
        current_time_frame = time_frames_to_eval[i]
        total_rel_error_for_frame = 0.0

        # Loop over the test dataset in batches to calculate error for the current time frame
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Get model prediction (full time series)
            pred_batch = model(x_batch)

            # De-normalize both prediction and ground truth to compute error in physical space
            pred_batch_decoded = dataset.normalizer_y.decode(pred_batch)
            y_batch_decoded = dataset.normalizer_y.decode(y_batch)

            # Extract the specific time slice for the current frame
            pred_t = pred_batch_decoded[..., time_idx]
            y_t = y_batch_decoded[..., time_idx]

            # Calculate the sum of relative L2 errors for the batch
            # LpLoss(size_average=False) returns the sum of ||pred-true||/||true||
            batch_error = l2_loss_criterion(pred_t, y_t).item()
            total_rel_error_for_frame += batch_error

        # Calculate the average error for this frame across all test samples
        avg_error = total_rel_error_for_frame / len(test_dataset)
        avg_errors_per_frame.append(avg_error)
        print(f"Time Frame: {current_time_frame}, Avg. Rel. L2 Error: {avg_error:.6f}")

print("Evaluation finished.")


# ============================================================================
# 4. PLOT AND SAVE RESULTS
# ============================================================================
print("\nPlotting and saving results...")

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(time_frames_to_eval, avg_errors_per_frame, marker='o', linestyle='-', color='b', label=f'{approach.capitalize()} Model')
plt.title(f'Relative L2-Error vs. Time Frames for {case_study}', fontsize=16)
plt.xlabel('Time Frame', fontsize=12)
plt.ylabel('Average Relative L2-Error', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

# --- Save the Plot ---
output_dir = 'OOD_Plots'
os.makedirs(output_dir, exist_ok=True)
plot_filename = f'{case_study}_{approach}_error_vs_time.png'
plot_filepath = os.path.join(output_dir, plot_filename)
plt.savefig(plot_filepath)
print(f"Plot saved to: {plot_filepath}")
plt.show()

# --- Save the Data to a .mat File ---
mat_output_dir = 'OOD_MatFiles'
os.makedirs(mat_output_dir, exist_ok=True)
mat_filename = f'{case_study}_{approach}_error_vs_time.mat'
mat_filepath = os.path.join(mat_output_dir, mat_filename)

# Create a dictionary with the results for saving
results_to_save = {
    'time_frames': time_frames_to_eval,
    'rel_l2_error': np.array(avg_errors_per_frame),
    'case_study': case_study,
    'approach': approach,
    'model_path': model_path
}

savemat(mat_filepath, results_to_save)
print(f"Error data saved to .mat file: {mat_filepath}")