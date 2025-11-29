import torch
import numpy as np
import importlib
import os
from utilities import ImportDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.colors import LightSource
from scipy.io import savemat

# ============================================================================
# 1. CHOOSE CASE STUDY AND MODEL APPROACH
# ============================================================================
case_study = 'AC3D'    # Options: 'SH3D', 'AC3D', 'CH3D', 'MBE3D', 'PFC3D'
approach = 'hybrid'  # Options: 'standard', 'hybrid'  <-- YOUR NEW CHOICE

# Validate the approach choice
if approach not in ['standard', 'hybrid']:
    raise ValueError(f"Unknown approach: '{approach}'. Please choose 'standard' or 'hybrid'.")

# Determine initial_condition_type based on the case_study
if case_study in ['SH3D', 'AC3D']:
    initial_condition_type = 'sphere'
elif case_study in ['CH3D', 'PFC3D']:
    initial_condition_type = 'star'
elif case_study == 'MBE3D':
    initial_condition_type = 'torus'
else:
    # A default or an error for unhandled cases
    raise ValueError(f"Unknown case_study: {case_study}. Please choose from 'SH3D', 'AC3D', 'CH3D', 'MBE3D', 'PFC3D'.")

print(f"Running Case Study: {case_study.upper()}")
print(f"Selected Model Approach: {approach.upper()}")
print(f"Selected Initial Condition: {initial_condition_type.upper()}")


# ============================================================================
# 2. MODEL AND DATASET LOADING
# ============================================================================
# Load model
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Define model paths for each case study
model_paths = {
    'AC3D': {
        'standard': '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'hybrid': '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
    },
    'CH3D': {
        'standard': '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'hybrid': '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
    },
    'SH3D': {
        'standard': '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'hybrid': '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
    },
    'MBE3D': {
        'standard': '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'hybrid': '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
    },
    'PFC3D': {
        'standard': '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt',
        'hybrid': '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
    }
}

# Automatically select the model path based on the chosen case_study and approach
model_path = model_paths[case_study][approach] # <-- CHANGED: Using 'approach' variable
print(f"Loading model: {model_path}")


try:
    from networks import TNO3d # Assuming networks.py and TNO3d are available

    # Add builtins.set to safe globals for robust loading
    torch.serialization.add_safe_globals([set])
    torch.serialization.add_safe_globals([TNO3d]) # Add TNO3d if it's part of the global scope during saving

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
except Exception as e:
    print(f"Secure loading failed: {e}\nFalling back to weights_only=False")
    # It's good practice to ensure the safe globals are added even for fallback
    torch.serialization.add_safe_globals([set])
    torch.serialization.add_safe_globals([TNO3d])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)


model = checkpoint['model']
model.eval()

# Load dataset for normalization AND EXTRACT PROBLEM NAME
# The 'problem' variable is now the same as 'case_study'
problem = case_study
print(f"Problem Name Determined: {problem}")

network_name = 'TNO3d'
cf = importlib.import_module(f"configs.config_{problem}_{network_name}") # Assuming configs module is available
dataset = ImportDataset(cf.parent_dir, cf.matlab_dataset, cf.normalized, cf.T_in, cf.T_out)


dataset.normalizer_x.mean = dataset.normalizer_x.mean.to(device)
dataset.normalizer_x.std = dataset.normalizer_x.std.to(device)
dataset.normalizer_y.mean = dataset.normalizer_y.mean.to(device)
dataset.normalizer_y.std = dataset.normalizer_y.std.to(device)


# ============================================================================
# 3. CREATE INITIAL CONDITION
# ============================================================================
def create_initial_condition(ic_type, case_study):

    Nx, Ny, Nz = 32, 32, 32
    Nt = 100
    u = None

    if ic_type == 'sphere':
        if case_study == 'AC3D':
            Lx = 5
            epsilon = 0.1
            dt = 0.005
            selected_frames = [0, 50, 90]
            radius = 0.5
        elif case_study == 'SH3D':
            Lx = 15
            epsilon = 0.15
            dt = 0.05
            selected_frames = [0, 70, 90]
            radius = 2.0
        else:
            raise ValueError(f"Sphere IC is not configured for case study: {case_study}")

        Ly, Lz = Lx, Lx
        x_grid = np.linspace(-Lx / 2, Lx / 2, Nx)
        y_grid = np.linspace(-Ly / 2, Ly / 2, Ny)
        z_grid = np.linspace(-Lz / 2, Lz / 2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        interface_width = np.sqrt(2) * epsilon
        u = np.tanh((radius - np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)) / interface_width)

    elif ic_type == 'star':
        selected_frames = [0, 50, 90]
        dt = 0.005
        if case_study == 'AC3D' or case_study == 'CH3D': # Assuming CH3D uses similar parameters to AC3D
            Lx = 2 if case_study == 'CH3D' else 5 # Use Lx=5 for AC3D, Lx=2 for CH3D
            epsilon = 0.05
            R_theta_func = lambda theta: 0.7 + 0.2 * np.cos(6 * theta)
        elif case_study == 'PFC3D':
            Lx = 10 * np.pi
            epsilon = 0.5
            R_theta_func = lambda theta: 5.0 + 1.0 * np.cos(6 * theta)
        else:
            raise ValueError(f"Star IC is not configured for case study: {case_study}")

        Ly, Lz = Lx, Lx
        x_grid = np.linspace(-Lx / 2, Lx / 2, Nx)
        y_grid = np.linspace(-Ly / 2, Ly / 2, Ny)
        z_grid = np.linspace(-Lz / 2, Lz / 2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        interface_width = np.sqrt(2.0) * epsilon
        theta = np.arctan2(zz, xx)
        R_theta = R_theta_func(theta)
        dist = np.sqrt(xx ** 2 + 2 * yy ** 2 + zz ** 2)
        u = np.tanh((R_theta - dist) / interface_width)

    elif ic_type == 'torus':
        if case_study == 'MBE3D':
            Lx = 2 * np.pi
            epsilon = 0.1
            dt = 0.001
            selected_frames = [0, 50, 90]
            R_major, r_minor = 2.1, 0.7
        else:
            raise ValueError(f"Torus IC is not configured for case study: {case_study}")

        Ly, Lz = Lx, Lx
        x_grid = np.linspace(-Lx / 2, Lx / 2, Nx)
        y_grid = np.linspace(-Ly / 2, Ly / 2, Ny)
        z_grid = np.linspace(-Lz / 2, Lz / 2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        interface_width = np.sqrt(2) * epsilon
        torus_dist = np.sqrt((np.sqrt(xx ** 2 + yy ** 2) - R_major) ** 2 + zz ** 2)
        u = np.tanh((r_minor - torus_dist) / interface_width)

    # Keep other initial conditions for completeness, though they are not triggered by the new logic
    elif ic_type == 'dumbbell':
        Lx, Ly, Lz = 40, 20, 20
        epsilon = 0.005
        dt = 0.01
        selected_frames = [0, 50, 90]
        x_grid, y_grid, z_grid = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny), np.linspace(0, Lz, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        R0, interface_width = 0.25, np.sqrt(2) * epsilon
        r1 = np.sqrt((xx - (0.3 * Lx)) ** 2 + (yy - (0.5 * Ly)) ** 2 + (zz - (0.5 * Lz)) ** 2)
        r2 = np.sqrt((xx - (1.7 * Lx)) ** 2 + (yy - (0.5 * Ly)) ** 2 + (zz - (0.5 * Lz)) ** 2)
        u_spheres = np.tanh((R0 - r1) / interface_width) + np.tanh((R0 - r2) / interface_width) + 1
        bar_mask = (xx > (0.4 * Lx)) & (xx < (1.6 * Lx)) & \
                   (yy > (0.4 * Ly)) & (yy < (0.6 * Ly)) & \
                   (zz > (0.4 * Lz)) & (zz < (0.6 * Lz))
        u = u_spheres; u[bar_mask] = 1.0; u = np.clip(u, -1.0, 1.0)

    elif ic_type == 'separation':
        Lx, Ly, Lz = 2 * np.pi, 2 * np.pi, 2 * np.pi
        epsilon = 0.5
        dt = 0.0005
        selected_frames = [0, 50, 90]
        x_grid, y_grid, z_grid = np.linspace(-Lx/2, Lx/2, Nx), np.linspace(-Ly/2, Ly/2, Ny), np.linspace(-Lz/2, Lz/2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        interface_width = np.sqrt(2) * epsilon
        r1_dist = np.sqrt((xx + 1)**2 + yy**2 + zz**2)
        r2_dist = np.sqrt((xx - 1)**2 + yy**2 + zz**2)
        u = np.tanh((1 - r1_dist) / interface_width) + np.tanh((1 - r2_dist) / interface_width)

    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")

    return u, (Lx, Ly, Lz), (Nx, Ny, Nz), Nt, dt, selected_frames

# Create the selected initial condition
initial_condition_field, domain_lengths, grid_sizes, Nt, dt, selected_frames = create_initial_condition(
    ic_type=initial_condition_type, case_study=case_study)
Lx, Ly, Lz = domain_lengths
Nx, Ny, Nz = grid_sizes

# ============================================================================
# 4. PREDICTION AND VISUALIZATION
# ============================================================================
# Prepare tensor for the model
input_tensor = torch.from_numpy(initial_condition_field).float().unsqueeze(0).unsqueeze(-1).to(device)
input_tensor = dataset.normalizer_x.encode(input_tensor)

# Run prediction
with torch.no_grad():
    # Start timer
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()  # Wait for all operations to complete
    start_time.record()  # Start recording

    prediction = model(input_tensor)

    end_time.record()  # Stop recording
    torch.cuda.synchronize()  # Wait for all operations to complete

    # Calculate elapsed time in milliseconds
    inference_time_ms = start_time.elapsed_time(end_time)
    # Convert milliseconds to seconds for saving
    inference_time = inference_time_ms / 1000.0

    prediction = dataset.normalizer_y.decode(prediction)
    # Clip the prediction to be within the physical bounds [-1, 1].
    #prediction = torch.clamp(prediction, min=-1.0, max=1.0)

# Updated print statement to show both units
print(f"Inference time: {inference_time_ms:.3f} milliseconds ({inference_time:.6f} seconds)")


# Create figure
fig = plt.figure(figsize=(20, 10))
grid = plt.GridSpec(2, len(selected_frames), hspace=0.3, wspace=0.2)
# The title now also includes the approach
fig.suptitle(f"TNO Prediction ({approach.capitalize()}) for {initial_condition_type.upper()} IC ({problem})", fontsize=16)

# Define mesh coordinates for plotting
if initial_condition_type in ['sphere', 'star', 'torus', 'separation']:
    x_coords = np.linspace(-Lx / 2, Lx / 2, Nx)
    x_lim_low, x_lim_high = -Lx / 2, Lx / 2
    y_lim_low, y_lim_high = -Ly / 2, Ly / 2
    z_lim_low, z_lim_high = -Lz / 2, Lz / 2
elif initial_condition_type == 'dumbbell':
    x_coords = np.linspace(0, Lx, Nx)
    x_lim_low, x_lim_high = 0, Lx
    y_lim_low, y_lim_high = 0, Ly
    z_lim_low, z_lim_high = 0, Lz

# 1. Plot 3D isosurfaces
for i, t in enumerate(selected_frames):
    ax = fig.add_subplot(grid[0, i], projection='3d')

    frame_data = prediction[0, ..., t].cpu().numpy()
    title_text = f'Prediction\nTime = {t}'


    level = 0.0
    try:
        dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)
        verts, faces, _, _ = measure.marching_cubes(frame_data, level=level, spacing=(dx, dy, dz))

        if initial_condition_type in ['sphere', 'star', 'torus', 'separation']:
            verts[:, 0] -= Lx / 2
            verts[:, 1] -= Ly / 2
            verts[:, 2] -= Lz / 2

        ls = LightSource(azdeg=135, altdeg=45)
        mesh = Poly3DCollection(verts[faces], facecolors='gray', edgecolor='none', alpha=0.9)
        ax.add_collection3d(mesh)
        plot_success = True
    except ValueError as e:
        print(f"Could not generate isosurface for frame {t}: {e}")
        ax.text(0.5, 0.5, 0.5, f"No isosurface\nat level={level:.2f}", ha='center', va='center', transform=ax.transAxes)
        plot_success = False

    ax.set_xlim(x_lim_low, x_lim_high)
    ax.set_ylim(y_lim_low, y_lim_high)
    ax.set_zlim(z_lim_low, z_lim_high)
    ax.set_title(f'{title_text}\nLevel = {level:.2f}', pad=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if plot_success:
        ax.view_init(elev=30, azim=45)

# 2. Plot 1D profile
ax_profile = fig.add_subplot(grid[1, :])
center_y_idx = Ny // 2
center_z_idx = Nz // 2

for t in selected_frames:
    if t == 0:
        profile = initial_condition_field[:, center_y_idx, center_z_idx]
        label_text = f't={t} (IC)'
    else:
        profile = prediction[0, :, center_y_idx, center_z_idx, t].cpu().numpy()
        label_text = f't={t}'

    ax_profile.plot(x_coords, profile, label=label_text, alpha=0.8, linewidth=1.5)

ax_profile.set_xlabel('Position along x-axis', fontsize=12)
ax_profile.set_ylabel('Field value', fontsize=12)
ax_profile.set_title('1D Profile Evolution Through Domain Center', pad=15)
ax_profile.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_profile.grid(True, alpha=0.3)
ax_profile.set_ylim([-1.5, 1.5])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


## This is saved the data of initial condition itself

# ============================================================================
# 5. SAVE RESULTS TO .MAT FILE
# ============================================================================
# Get the raw prediction tensor from the model as a NumPy array
prediction_np = prediction.cpu().numpy()
# Create a "hybrid" tensor for saving, ensuring the t=0 slice is the true IC
final_prediction_to_save = np.copy(prediction_np)
final_prediction_to_save[0, :, :, :, 0] = initial_condition_field

# Define the output filename based on whether the model is a hybrid version.
# Suffix is PI-MHNO for hybrid models, MHNO otherwise.
if 'Hybrid' in model_path:
    suffix = '_PI-MHNO'
else:
    suffix = '_MHNO'
base_filename = f'{problem}_python_predictions_{initial_condition_type}{suffix}.mat'

# Define the output directory and create it if it doesn't exist.
output_dir = 'OOD_MatFiles'
os.makedirs(output_dir, exist_ok=True)

# Combine directory and filename for the full save path.
output_filepath = os.path.join(output_dir, base_filename)

# Save the corrected data and the inference time to the .mat file
savemat(output_filepath, {
    'python_pred': final_prediction_to_save,
    'selected_frames': np.array(selected_frames),
    'x': x_coords,
    'inference_time': inference_time  # Add the inference time (in seconds)
})
print(f"Corrected prediction (with true IC at t=0) and inference time saved to {output_filepath}")