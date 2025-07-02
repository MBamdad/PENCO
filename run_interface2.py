import torch
import numpy as np
import importlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# --- Model and Normalizer Loading (Unchanged) ---
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_path = '/scratch/noqu8762/phase_field_equations_4d/SH3D/models_smooth/TNO3d_SH3D_S32_T1to100_width12_modes16_q12_h6.pt'

try:
    from networks import TNO3d
    torch.serialization.add_safe_globals([TNO3d])
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
except Exception as e:
    print(f"Secure loading failed: {e}\nFalling back to weights_only=False")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

model = checkpoint['model']
model.eval()

problem = 'SH3D'
network_name = 'TNO3d'
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")

# 1. Load the actual dataset the model was trained/tested on
dataset_path = './data/SH3D_1500_Nt_101_Nx_32_random.pt'
print(f"Loading ground truth dataset from: {dataset_path}")
full_dataset = torch.load(dataset_path)

if isinstance(full_dataset, dict) and 'data' in full_dataset:
    data_tensor = full_dataset['data']
else:
    raise ValueError("Loaded dataset is not in the expected format.")

print(f"Original data tensor shape: {data_tensor.shape}")
permuted_tensor = data_tensor.permute(0, 2, 3, 4, 1)
print(f"Permuted data tensor shape: {permuted_tensor.shape}")

data_x = permuted_tensor[..., 0:1]
data_y = permuted_tensor[..., 1:]

print(f"Split into input 'x' shape: {data_x.shape}")
print(f"Split into output 'y' shape: {data_y.shape}")

# 2. Pick a single sample to test
sample_idx = 10
print(f"Using sample index: {sample_idx}")

# 3. Separate the input and the ground truth for our chosen sample
input_ic = data_x[sample_idx].unsqueeze(0).to(device)
ground_truth_evolution = data_y[sample_idx].to(device)

# IMPORTANT: We still need the normalizer
dataset_loader = importlib.import_module('utilities').ImportDataset
normalizer = dataset_loader(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out).normalizer_x
normalizer.mean = normalizer.mean.to(device)
normalizer.std = normalizer.std.to(device)

# Normalize the input before feeding it to the model
normalized_input = normalizer.encode(input_ic)

# 4. Run the model to get the prediction
with torch.no_grad():
    prediction_normalized = model(normalized_input)
    prediction = normalizer.decode(prediction_normalized)

# Squeeze the batch dimension for easier plotting
prediction = prediction.squeeze(0)

# 5. Compare the prediction with the ground truth
# --- MODIFIED: AVOID THE LAST FRAME (t=100) AS IT IS CORRUPT IN THE DATA FILE ---
# We will plot t=99 (model index 98) instead.
selected_frames_model_idx = [0, 49, 98]
selected_frames_true_time = [1, 50, 99]

fig = plt.figure(figsize=(15, 8))
fig.suptitle(f'Model Validation on Sample {sample_idx}', fontsize=16)

# Plot Ground Truth
for i, t_idx in enumerate(selected_frames_model_idx):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    frame_data = ground_truth_evolution[..., t_idx].cpu().numpy()
    print(f"Ground Truth (t={selected_frames_true_time[i]}): min={np.min(frame_data):.4f}, max={np.max(frame_data):.4f}")
    try:
        verts, faces, _, _ = measure.marching_cubes(frame_data, level=0.0)
        mesh = Poly3DCollection(verts[faces], facecolors='blue', edgecolor='none', alpha=0.9)
        ax.add_collection3d(mesh)
        ax.set_title(f'Ground Truth (t={selected_frames_true_time[i]})')
    except (ValueError, RuntimeError):
        ax.set_title(f'Ground Truth (t={selected_frames_true_time[i]})\n(No Isosurface)')
    ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=30, azim=45)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

# Plot Model Prediction
for i, t_idx in enumerate(selected_frames_model_idx):
    ax = fig.add_subplot(2, 3, i + 4, projection='3d')
    frame_data = prediction[..., t_idx].cpu().numpy()
    print(f"Model Prediction (t={selected_frames_true_time[i]}): min={np.min(frame_data):.4f}, max={np.max(frame_data):.4f}")
    try:
        verts, faces, _, _ = measure.marching_cubes(frame_data, level=0.0)
        mesh = Poly3DCollection(verts[faces], facecolors='red', edgecolor='none', alpha=0.9)
        ax.add_collection3d(mesh)
        ax.set_title(f'Model Prediction (t={selected_frames_true_time[i]})')
    except (ValueError, RuntimeError):
        ax.set_title(f'Model Prediction (t={selected_frames_true_time[i]})\n(No Isosurface)')
    ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=30, azim=45)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()