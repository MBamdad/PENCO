import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib
matplotlib.use('TkAgg')
# --- 0. Setup Output Directory ---
plot_dir = 'DNS_Results'
os.makedirs(plot_dir, exist_ok=True)
print(f"Figures will be saved in the '{plot_dir}' directory.")

# --- 1. Parameter Initialization (Matching MATLAB Script) ---

# Spatial Parameters
Nx = 32  # Grid size in x direction
Ny = Nx  # Grid size in y direction
Nz = Nx  # Grid size in z direction
Lx = 10.0  # Domain size in x direction
Ly = 10.0  # Domain size in y direction
Lz = 10.0  # Domain size in z direction
hx = Lx / Nx
hy = Ly / Ny
hz = Lz / Nz

# Create the grid
x = np.linspace(-0.5 * Lx, 0.5 * Lx - hx, Nx)
y = np.linspace(-0.5 * Ly, 0.5 * Ly - hy, Ny)
z = np.linspace(-0.5 * Lz, 0.5 * Lz - hz, Nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# Constant for the PDE
epsilon_pde = 0.15

# Discrete Fourier Transform Setup (using np.fft.fftfreq for precision)
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=hx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=hy)
kz = 2 * np.pi * np.fft.fftfreq(Nz, d=hz)
kxx, kyy, kzz = np.meshgrid(kx ** 2, ky ** 2, kz ** 2, indexing='ij')

# Time Discretization
dt = 0.05  # Time step
Nt = 100  # Total number of time steps

# --- 2. Simulation for a Single, Specific Initial Condition ---
print('Starting direct numerical simulation for a single sphere...')

# Create the specific SMOOTH sphere initial condition
radius = 2.0
epsilon_interface = 0.1
distance = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
u = np.tanh((radius - distance) / epsilon_interface)

# Initialize storage for solutions at specific time frames
u_t0 = u.copy()  # Store initial condition (Time Frame 0)
u_t50 = np.zeros((Nx, Ny, Nz))
u_t100 = np.zeros((Nx, Ny, Nz))

# --- 3. Time Evolution Loop (Directly solving the PDE) ---
for iter in range(1, Nt + 1):
    # Time evolution of the SH3D equation using a spectral method
    u_hat = np.fft.fftn(u)
    u3_hat = np.fft.fftn(u ** 3)

    # Equation in Fourier space
    s_hat = u_hat / dt - u3_hat + 2 * (kxx + kyy + kzz) * u_hat
    v_hat = s_hat / (1.0 / dt + (1 - epsilon_pde) + (kxx + kyy + kzz) ** 2)

    # Transform back to real space
    u = np.real(np.fft.ifftn(v_hat))

    # Store solutions at specified time frames
    if iter == 50:
        u_t50 = u.copy()
        print('Captured data at Time Frame 50.')
    elif iter == 100:
        u_t100 = u.copy()
        print('Captured data at Time Frame 100.')

print('Simulation complete.')

# --- 4. Plotting Results (Replicating MATLAB Style) ---

# Set a professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# --- Plot 1: 3D Subplot of Phase Evolution ---
print("Generating 3D evolution plot...")
fig_3d = plt.figure(figsize=(20, 6))
time_data = [(u_t0, 'Time Frame: 0', '#0072BD'),
             (u_t50, 'Time Frame: 50', '#D95319'),
             (u_t100, 'Time Frame: 100', '#77AC30')]

for i, (data, title, color) in enumerate(time_data):
    ax = fig_3d.add_subplot(1, 3, i + 1, projection='3d')

    try:
        verts, faces, _, _ = measure.marching_cubes(data, level=0.0, spacing=(hx, hy, hz))
        verts -= np.array([Lx / 2, Ly / 2, Lz / 2])

        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor(color)
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim([-Lx / 2, Lx / 2])
        ax.set_ylim([-Ly / 2, Ly / 2])
        ax.set_zlim([-Lx / 2, Lz / 2])

        ax.set_box_aspect([Lx, Ly, Lz])

        ax.set_title(title)
        ax.view_init(elev=30, azim=45)
        ax.set_proj_type('ortho')

    except (ValueError, RuntimeError) as e:
        ax.text(0, 0, 0, "No isosurface found", ha='center', va='center')
        ax.set_title(title)
        print(f"Could not generate isosurface for '{title}': {e}")

fig_3d.tight_layout()

# <<< --- SAVE THE 3D FIGURE --- >>>
save_path_3d = os.path.join(plot_dir, 'SH3D_DNS_Evolution_3D.png')
fig_3d.savefig(save_path_3d, dpi=300, bbox_inches='tight')
print(f"3D plot saved to: {save_path_3d}")
# <<< ------------------------ >>>

plt.show()

# --- Plot 2: 1D Profile Plot ---
print("Generating 1D profile plot...")
fig_1d, ax_1d = plt.subplots(figsize=(12, 7))

# Extract 1D profile along the x-axis (where y=0 and z=0)
mid_idx_y = Nx // 2
mid_idx_z = Nz // 2

profile_t0 = u_t0[:, mid_idx_y, mid_idx_z]
profile_t50 = u_t50[:, mid_idx_y, mid_idx_z]
profile_t100 = u_t100[:, mid_idx_y, mid_idx_z]

# Plotting with styles matching MATLAB
ax_1d.plot(x, profile_t0, color='#0072BD', linewidth=2.5, label='Time Frame: 0')
ax_1d.plot(x, profile_t50, color='#D95319', linewidth=2.5, label='Time Frame: 50')
ax_1d.plot(x, profile_t100, color='#EDB120', linewidth=2.5, label='Time Frame: 100')

ax_1d.set_title('1D Profile of Phase Field u along x-axis (y=0, z=0)')
ax_1d.set_xlabel('x')
ax_1d.set_ylabel('u(x, y=0, z=0)')
ax_1d.legend(loc='best')
ax_1d.set_ylim([-1.1, 1.1])
ax_1d.grid(True, which='both', linestyle='--', linewidth=0.7)

# <<< --- SAVE THE 1D FIGURE --- >>>
save_path_1d = os.path.join(plot_dir, 'SH3D_DNS_Profile_1D.png')
fig_1d.savefig(save_path_1d, dpi=300, bbox_inches='tight')
print(f"1D plot saved to: {save_path_1d}")
# <<< ------------------------ >>>

plt.show()