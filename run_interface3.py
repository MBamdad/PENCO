import torch
import numpy as np
import importlib
import os
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' for non-GUI environments
import matplotlib.pyplot as plt
from scipy.io import savemat

from utilities import ImportDataset
from torch.utils.data import DataLoader, random_split

# ============================================================================
# 1. CHOOSE CASE STUDY AND MODEL APPROACH
# ============================================================================
case_study = 'AC3D'
approach = 'MHNO' # You can now switch this to 'MHNO' or 'FNO4d' and PI_MHNO

if approach not in ['MHNO', 'PI_MHNO', 'FNO4d']:
    raise ValueError(f"Unknown approach: '{approach}'.")

print(f"Running Case Study: {case_study.upper()}")
print(f"Selected Model Approach: {approach.upper()}")

# ============================================================================
# 2. MODEL AND DATASET LOADING
# ============================================================================
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_paths = {
    # ... (your model_paths dictionary remains unchanged) ...
    'AC3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d': '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/FNO4d_AC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'CH3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d': '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/FNO4d_CH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'SH3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d': '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/FNO4d_SH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'MBE3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d': '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/FNO4d_MBE3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    },
    'PFC3D': {
        'MHNO': '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'PI_MHNO': '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'FNO4d': '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/FNO4d_PFC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt'
    }
}


model_path = model_paths[case_study][approach]
print(f"Loading model: {model_path}")

# ======================= START: MODIFIED LOADING LOGIC ========================

# Step 1: Determine network type and load its config file
if 'TNO3d' in model_path:
    network_name = 'TNO3d'
elif 'FNO4d' in model_path:
    network_name = 'FNO4d'
else:
    raise ValueError("Could not determine network type from model path.")

problem = case_study
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")

# Step 2: Instantiate the model architecture
print(f"Instantiating a '{network_name}' model structure.")
network_class = getattr(importlib.import_module('networks'), network_name)

if network_name == 'TNO3d':
    model = network_class(
        modes1=cf.modes, modes2=cf.modes, modes3=cf.modes,
        width=cf.width, width_q=cf.width_q, width_h=cf.width_h,
        T_in=cf.T_in, T_out=cf.T_out, n_layers=cf.n_layers
    ).to(device)
elif network_name == 'FNO4d':
    model = network_class(
        modes1=cf.modes, modes2=cf.modes, modes3=cf.modes,
        modes4_internal=1, width=cf.width, width_q=cf.width_q,
        T_in_channels=cf.T_in, n_layers=cf.n_layers
    ).to(device)

# Step 3: Load the checkpoint and check for keys
checkpoint = torch.load(model_path, map_location=device)

# Step 4: Load the weights using the correct key
if 'model_state_dict' in checkpoint:
    print("Found 'model_state_dict' key. Loading weights into the model.")
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'model' in checkpoint:
    print("Found 'model' key. Overwriting instantiated model with the saved object.")
    model = checkpoint['model']
    model.to(device) # Ensure it's on the correct device
else:
    raise KeyError("Checkpoint is in an unknown format. Neither 'model' nor 'model_state_dict' found.")

model.eval()
print("Model loaded and in evaluation mode.")

# ======================== END: MODIFIED LOADING LOGIC =========================

# Load dataset (config 'cf' is already loaded)
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)

dataset = ImportDataset(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out)
if cf.normalized:
    dataset.normalizer_x.to(device)
    dataset.normalizer_y.to(device)

train_size, test_size = cf.nTrain, cf.nTest
if train_size + test_size > len(dataset):
    test_size = len(dataset) - train_size
train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, len(dataset) - train_size - test_size])
test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False)
print(f"Loaded and split dataset. Using {len(test_dataset)} samples for testing.")


# ============================================================================
# 3. EVALUATION ON TEST DATA AND STATS CALCULATION (No changes needed here)
# ============================================================================
# ... (The rest of your script from line 113 onwards remains exactly the same) ...

start_frame, end_frame = 0, 90
time_frames_to_eval = np.arange(start_frame, end_frame + 1)
output_indices = time_frames_to_eval - cf.T_in - 1
valid_mask = (output_indices >= 0) & (output_indices < cf.T_out)
time_frames_to_eval = time_frames_to_eval[valid_mask]
output_indices = output_indices[valid_mask]

print(f"\nEvaluating on time frames: {time_frames_to_eval[0]} to {time_frames_to_eval[-1]}")
stats = {'mean': [], 'median': [], 'q1': [], 'q3': [], 'std': []}

print("\nStarting evaluation on the test set...")
model.eval()
with torch.no_grad():
    all_preds_decoded = []
    all_truth_decoded = []
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred_batch = model(x_batch)
        all_preds_decoded.append(dataset.normalizer_y.decode(pred_batch).cpu())
        all_truth_decoded.append(dataset.normalizer_y.decode(y_batch).cpu())

    all_preds_decoded = torch.cat(all_preds_decoded, dim=0)
    all_truth_decoded = torch.cat(all_truth_decoded, dim=0)

    for i, time_idx in enumerate(output_indices):
        current_time_frame = time_frames_to_eval[i]
        pred_t = all_preds_decoded[..., time_idx]
        y_t = all_truth_decoded[..., time_idx]
        numerator = torch.norm(pred_t.reshape(pred_t.shape[0], -1) - y_t.reshape(y_t.shape[0], -1), p=2, dim=1)
        denominator = torch.norm(y_t.reshape(y_t.shape[0], -1), p=2, dim=1)
        rel_errors = (numerator / (denominator + 1e-9)).numpy()
        stats['mean'].append(np.mean(rel_errors))
        stats['median'].append(np.median(rel_errors))
        stats['q1'].append(np.percentile(rel_errors, 25))
        stats['q3'].append(np.percentile(rel_errors, 75))
        stats['std'].append(np.std(rel_errors))
        print(f"Time Frame: {current_time_frame}, Mean Rel. L2 Error: {stats['mean'][-1]:.6f}")

print("Evaluation finished.")

# ============================================================================
# 4. PLOT AND SAVE RESULTS
# ============================================================================
print("\nPlotting and saving results...")

# Convert lists to numpy arrays for easier plotting
for key in stats:
    stats[key] = np.array(stats[key])

# Create the plot
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot shaded area for +- 1 Std Dev
ax.fill_between(time_frames_to_eval,
                stats['mean'] - stats['std'],
                stats['mean'] + stats['std'],
                color='purple', alpha=0.15, label='±1 Std Dev')

# Plot lines
ax.plot(time_frames_to_eval, stats['q1'], 'r--', label='Q1', linewidth=1.5)
ax.plot(time_frames_to_eval, stats['median'], 'b-', label='Median', linewidth=1.5)
ax.plot(time_frames_to_eval, stats['q3'], 'g--', label='Q3', linewidth=1.5)
ax.plot(time_frames_to_eval, stats['mean'], 'k-', label='Mean', linewidth=2)

ax.set_yscale('log')
ax.set_title(f'L2 Error Statistics for {case_study} ({approach})', fontsize=16)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('L2 Error', fontsize=12)
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
ax.legend(loc='best')
ax.set_xlim(left=0, right=90)
plt.tight_layout()

# --- Save the Plot ---
output_dir = 'OOD_Plots_Stats'
os.makedirs(output_dir, exist_ok=True)
plot_filename = f'{case_study}_{approach}_error_stats.png'
plot_filepath = os.path.join(output_dir, plot_filename)
plt.savefig(plot_filepath, dpi=300)
print(f"Plot saved to: {plot_filepath}")
plt.show()

# --- Save the Data to a .mat File ---
mat_output_dir = 'OOD_MatFiles_Stats'
os.makedirs(mat_output_dir, exist_ok=True)
mat_filename = f'{case_study}_{approach}_error_stats.mat'
mat_filepath = os.path.join(mat_output_dir, mat_filename)

results_to_save = {
    'time_steps': time_frames_to_eval,
    'mean_error': stats['mean'],
    'median_error': stats['median'],
    'q1_error': stats['q1'],
    'q3_error': stats['q3'],
    'std_dev_error': stats['std'],
    'case_study': case_study,
    'approach': approach,
    'model_path': model_path
}

savemat(mat_filepath, results_to_save)
print(f"Statistics data saved to .mat file: {mat_filepath}")