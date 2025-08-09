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
from scipy.io import savemat

# ============================================================================
# 1. CHOOSE INITIAL CONDITION
# ============================================================================
# Options: 'sphere', 'dumbbell', 'star', separation, torus
initial_condition_type = 'torus' # <-- CHANGE THIS VALUE TO RUN A DIFFERENT SIMULATION
print(f"Running simulation for Initial Condition: {initial_condition_type.upper()}")

# ============================================================================
# 2. MODEL AND DATASET LOADING
# ============================================================================
# Load model
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# AC3D
#model_path = '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt' # AC3D
#model_path = '/scratch/noqu8762/phase_field_equations_4d/AC3D/models/TNO3d_AC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt' # AC3D

# CH3D
#model_path = '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
#model_path = '/scratch/noqu8762/phase_field_equations_4d/CH3D/models/TNO3d_CH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'

# SH3D
model_path = '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
#model_path = '/scratch/noqu8762/phase_field_equations_4d/SH3D/models/TNO3d_SH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'


#  MBE3d
# model_path = '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
#model_path = '/scratch/noqu8762/phase_field_equations_4d/MBE3D/models/TNO3d_MBE3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'

# PFC
#model_path = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
#model_path = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'

# PFC3D Mixed
#model_path = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_Mixed_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'
#model_path = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/models/TNO3d_PFC3D_Mixed_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt'

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
key_directory = 'phase_field_equations_4d'
problem = ''
try:
    parts = model_path.split('/')
    index = parts.index(key_directory)
    problem = parts[index + 1] # This gets the directory name (e.g., 'AC3D')
except (ValueError, IndexError):
    print(f"Could not automatically determine problem name. Set manually if needed.")

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
def create_initial_condition(ic_type='sphere'):

    Nx, Ny, Nz = 0, 0, 0
    Lx, Ly, Lz = 0, 0, 0
    epsilon = 0
    Nt = 0
    selected_frames = []
    u = None
    dt = 0

    if ic_type == 'sphere':
        Nx = 32; Ny = 32; Nz = 32
        #Lx = 10*np.pi; # PFC3D
        #Lx = 5 # AC3D
        Lx = 15  # SH3D
        Ly = Lx; Lz = Lx
        #epsilon = 0.5 # PFC3D
        epsilon = 0.15 # SH3d
        # dt = 0.0005
        dt = 0.05 # SH3D
        Nt = 100
        selected_frames = [0, 70, 90]

        x_grid = np.linspace(-Lx / 2, Lx / 2, Nx)
        y_grid = np.linspace(-Ly / 2, Ly / 2, Ny)
        z_grid = np.linspace(-Lz / 2, Lz / 2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        #radius = 6 # PFC3D
        #radius = 0.5 # AC3D
        radius = 2.0  # SH3D
        interface_width = np.sqrt(2) * epsilon
        u = np.tanh((radius - np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)) / interface_width)

    elif ic_type == 'dumbbell':
        Nx = 32; Ny = 32; Nz = 32
        Lx = 40; Ly = 20; Lz = 20
        epsilon = 0.005
        dt = 0.01
        Nt = 100
        selected_frames = [0, 50, 90]

        x_grid = np.linspace(0, Lx, Nx)
        y_grid = np.linspace(0, Ly, Ny)
        z_grid = np.linspace(0, Lz, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        R0 = 0.25
        interface_width = np.sqrt(2) * epsilon

        r1 = np.sqrt((xx - (0.3 * Lx)) ** 2 + (yy - (0.5 * Ly)) ** 2 + (zz - (0.5 * Lz)) ** 2)
        r2 = np.sqrt((xx - (1.7 * Lx)) ** 2 + (yy - (0.5 * Ly)) ** 2 + (zz - (0.5 * Lz)) ** 2)
        u_spheres = np.tanh((R0 - r1) / interface_width) + np.tanh((R0 - r2) / interface_width) + 1

        bar_mask = (xx > (0.4 * Lx)) & (xx < (1.6 * Lx)) & \
                   (yy > (0.4 * Ly)) & (yy < (0.6 * Ly)) & \
                   (zz > (0.4 * Lz)) & (zz < (0.6 * Lz))
        u = u_spheres
        u[bar_mask] = 1.0
        u = np.clip(u, -1.0, 1.0)

    elif ic_type == 'star':
        Nx = 32; Ny = 32; Nz = 32
        #Lx = 5 # AC3D,
        #Lx = 10 * np.pi --> PFC3D
        Lx = 2  # CH3D
        Ly = Lx; Lz = Lx
        #epsilon = 0.5 # PFC3D
        epsilon = 0.05 # AC3D
        dt = 0.005
        Nt = 100
        selected_frames = [0, 50, 90]

        x_grid = np.linspace(-Lx / 2, Lx / 2, Nx)
        y_grid = np.linspace(-Ly / 2, Ly / 2, Ny)
        z_grid = np.linspace(-Lz / 2, Lz / 2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        interface_width = np.sqrt(2.0) * epsilon
        theta = np.arctan2(zz, xx)
        #R_theta = 5.0 + 1.0 * np.cos(6 * theta) # PFC3D
        R_theta = 0.7 + 0.2 * np.cos(6 * theta)  # AC3D
        dist = np.sqrt(xx ** 2 + 2 * yy ** 2 + zz ** 2)
        u = np.tanh((R_theta - dist) / interface_width)

    elif ic_type == 'torus':
        Nx = 32; Ny = 32; Nz = 32
        #Lx = 10*np.pi;
        Lx = 2*np.pi # MBE3D
        Ly = Lx; Lz = Lx
        #epsilon = 0.5
        epsilon = 0.1 # MBE3D
        #dt = 0.005
        dt = 0.001 # MBE3D
        Nt = 100
        selected_frames = [0, 50, 90]

        x_grid = np.linspace(-Lx / 2, Lx / 2, Nx)
        y_grid = np.linspace(-Ly / 2, Ly / 2, Ny)
        z_grid = np.linspace(-Lz / 2, Lz / 2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        #R_major = 5.5
        #r_minor = 3.5
        R_major = 2.1 # MBE3D
        r_minor = 0.7 # MBE3D

        interface_width = np.sqrt(2) * epsilon
        torus_dist = np.sqrt((np.sqrt(xx ** 2 + yy ** 2) - R_major) ** 2 + zz ** 2)
        u = np.tanh((r_minor - torus_dist) / interface_width)
        #u = np.clip(u, -1.0, 1.0)

    elif ic_type == 'separation':
        Nx = 32; Ny = 32; Nz = 32
        Lx = 2 * np.pi; Ly = 2 * np.pi; Lz = 2 * np.pi
        epsilon = 0.5
        dt = 0.0005
        Nt = 100
        selected_frames = [0, 50, 90]

        x_grid = np.linspace(-Lx/2, Lx/2, Nx)
        y_grid = np.linspace(-Ly/2, Ly/2, Ny)
        z_grid = np.linspace(-Lz/2, Lz/2, Nz)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        interface_width = np.sqrt(2) * epsilon

        r1_dist = np.sqrt((xx + 1)**2 + yy**2 + zz**2)
        r2_dist = np.sqrt((xx - 1)**2 + yy**2 + zz**2)

        u = np.tanh((1 - r1_dist) / interface_width) + np.tanh((1 - r2_dist) / interface_width)
        #u = np.clip(u, -1.0, 1.0)

    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")

    return u, (Lx, Ly, Lz), (Nx, Ny, Nz), Nt, dt, selected_frames

# Create the selected initial condition
initial_condition_field, domain_lengths, grid_sizes, Nt, dt, selected_frames = create_initial_condition(
    ic_type=initial_condition_type)
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

    # ==================== MODIFICATION START ====================
    # Calculate elapsed time in milliseconds
    inference_time_ms = start_time.elapsed_time(end_time)
    # Convert milliseconds to seconds for saving
    inference_time = inference_time_ms / 1000.0
    # ===================== MODIFICATION END =====================

    prediction = dataset.normalizer_y.decode(prediction)
    # Clip the prediction to be within the physical bounds [-1, 1].
    #prediction = torch.clamp(prediction, min=-1.0, max=1.0)

# ==================== MODIFICATION START ====================
# Updated print statement to show both units
print(f"Inference time: {inference_time_ms:.3f} milliseconds ({inference_time:.6f} seconds)")
# ===================== MODIFICATION END =====================


# Create figure
fig = plt.figure(figsize=(20, 10))
grid = plt.GridSpec(2, len(selected_frames), hspace=0.3, wspace=0.2)
fig.suptitle(f"TNO Prediction for {initial_condition_type.upper()} Initial Condition ({problem})", fontsize=16)

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

    #if t == 0:
    #    frame_data = initial_condition_field
    #    title_text = f'Initial Condition\nTime = {t}'
    #else:
    #    frame_data = prediction[0, ..., t].cpu().numpy()
    #    title_text = f'Prediction\nTime = {t}'

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
    #profile = prediction[0, :, center_y_idx, center_z_idx, t].cpu().numpy()
    #label_text = f't={t}'

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
# Define the output filename using the 'problem' variable extracted earlier
# from the model_path. This makes the filename dynamic.
output_filename = f'{problem}_python_predictions_{initial_condition_type}.mat'

# ==================== MODIFICATION START ====================
# Save the corrected data and the inference time to the .mat file
savemat(output_filename, {
    'python_pred': final_prediction_to_save,
    'selected_frames': np.array(selected_frames),
    'x': x_coords,
    'inference_time': inference_time  # Add the inference time (in seconds)
})
print(f"Corrected prediction (with true IC at t=0) and inference time saved to {output_filename}")
# ===================== MODIFICATION END =====================