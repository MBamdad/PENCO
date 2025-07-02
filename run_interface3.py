import torch
import numpy as np
import importlib
from utilities import ImportDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.colors import LightSource

# Load model
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#model_path = '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6.pt'

model_path = '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'

# Option 1 (Recommended secure approach)
try:
    from networks import TNO3d  # Import your custom network class
    torch.serialization.add_safe_globals([TNO3d])
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
except Exception as e:
    print(f"Secure loading failed: {e}\nFalling back to weights_only=False")
    # Option 2 (Less secure fallback)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

model = checkpoint['model']
model.eval()

# Load dataset for normalization
problem = 'SH3D'
network_name = 'TNO3d'
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")
dataset = ImportDataset(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out)

# Move normalizer parameters to device
dataset.normalizer_x.mean = dataset.normalizer_x.mean.to(device)
dataset.normalizer_x.std = dataset.normalizer_x.std.to(device)
dataset.normalizer_y.mean = dataset.normalizer_y.mean.to(device)
dataset.normalizer_y.std = dataset.normalizer_y.std.to(device)


# Create spherical initial condition
def create_sharp_sphere_initial_condition(N=32, radius=2, L=10):
    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    z = np.linspace(-L / 2, L / 2, N)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    sphere = np.ones((N, N, N), dtype=np.float32)  # Use float32 for consistency

    # Perfectly sharp transition (no smoothing)
    outer_mask = r > radius
    sphere[outer_mask] = -1.0

    # Force exact values (no floating point artifacts)
    sphere = np.where(r <= radius, 1.0, -1.0)

    return sphere
# Create initial condition with perfect sharp interface
sphere_ic = create_sharp_sphere_initial_condition()

input_tensor = torch.from_numpy(sphere_ic).float().unsqueeze(0).unsqueeze(-1).to(device)
input_tensor = dataset.normalizer_x.encode(input_tensor)

# Run prediction
with torch.no_grad():
    prediction = model(input_tensor)  # Shape [1, 32, 32, 32, 10]
    prediction = dataset.normalizer_y.decode(prediction)

# Define your custom frames to display
selected_frames = [0, 50, 90]  # Adjusted for T_out=10
num_frames = len(selected_frames)

# Create figure with two subplots: 3D views and 1D profile
fig = plt.figure(figsize=(20, 10))
grid = plt.GridSpec(2, num_frames, hspace=0.3, wspace=0.2)

# 1. Plot 3D isosurfaces for selected frames
for i, t in enumerate(selected_frames):
    ax = fig.add_subplot(grid[0, i], projection='3d')
    frame_data = prediction[0, ..., t].cpu().numpy()

    # Print data range for debugging
    print(f"Frame {t}: min={np.min(frame_data):.3f}, max={np.max(frame_data):.3f}")

    # Determine appropriate level
    data_min, data_max = np.min(frame_data), np.max(frame_data)
    if data_min > 0 or data_max < 0:
        level = (data_max + data_min) / 2  # Midpoint if zero is outside range
    else:
        level = 0.0  # Default level

    try:
        # Extract smooth isosurface with adjusted level
        verts, faces, _, _ = measure.marching_cubes(frame_data, level=level)

        # Apply lighting and coloring
        ls = LightSource(azdeg=135, altdeg=45)
        rgb = ls.shade_normals(verts[faces], fraction=0.8)

        mesh = Poly3DCollection(verts[faces],
                                facecolors=rgb,
                                edgecolor='none',
                                alpha=0.9)

        ax.add_collection3d(mesh)
        plot_success = True
    except ValueError as e:
        print(f"Could not generate isosurface for frame {t}: {e}")
        plot_success = False
        # Display empty plot with error message
        ax.text(0.5, 0.5, 0.5, f"No isosurface\nat level={level:.2f}",
                ha='center', va='center', fontsize=10)

    # Set viewing parameters
    ax.set_xlim(0, frame_data.shape[0])
    ax.set_ylim(0, frame_data.shape[1])
    ax.set_zlim(0, frame_data.shape[2])
    ax.set_title(f'Time = {t}\nLevel = {level:.2f}', pad=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if plot_success:
        ax.view_init(elev=30, azim=45)

# 2. Plot 1D profile through center for all time steps
ax_profile = fig.add_subplot(grid[1, :])
L = 10  # Domain size
x = np.linspace(-L / 2, L / 2, prediction.shape[1])
center_idx = prediction.shape[1] // 2  # Middle of the domain

# Plot profiles for the same custom frames in the profile plot
for t in selected_frames:
    profile = prediction[0, :, center_idx, center_idx, t].cpu().numpy()
    ax_profile.plot(x, profile, label=f't={t}', alpha=0.8, linewidth=1.5)

# Format profile plot
ax_profile.set_xlabel('Position along x-axis', fontsize=12)
ax_profile.set_ylabel('Field value', fontsize=12)
ax_profile.set_title('1D Profile Evolution Through Domain Center', pad=15)
ax_profile.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_profile.grid(True, alpha=0.3)
ax_profile.set_ylim([-1.2, 1.5])  # Match MATLAB y-limits

plt.tight_layout()
plt.show()

# After getting predictions in Python
prediction_np = prediction.cpu().numpy()  # Convert to numpy array

# Save to .mat file
from scipy.io import savemat
savemat('SH3D_python_predictions.mat', {
    'python_pred': prediction_np,
    'selected_frames': np.array(selected_frames),
    'x': x  # Spatial coordinates
})