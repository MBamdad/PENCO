import os
import cv2
import vtk
import torch
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # or 'Agg', depending on your needs
import matplotlib.pyplot as plt
from vtk.util import numpy_support
# print(dir(numpy_support))
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.cm import ScalarMappable
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable


'''''
def plot_loss_trend(losses, labels, problem):
    folder = problem + "/plots"
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=(3.5, 4))
    for loss, label in zip(losses, labels):
        plt.semilogy(loss, label=label)
    plt.legend()
    os.makedirs(folder, exist_ok=True)
    plot_name = folder + "/LossTrend"
    plt.savefig(plot_name + '.png', dpi=600, bbox_inches='tight')
    plt.show()

    '''

'''''
def plot_loss_trend(losses, labels, problem, network_name, Final_time, test_l2_avg):
    folder = os.path.join(problem, f"plots_{network_name}")
    os.makedirs(folder, exist_ok=True)

    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font

    plt.figure(figsize=(3.5, 4))
    for loss, label in zip(losses, labels):
        plt.semilogy(loss, label=label)
    plt.legend()

    # Add annotation for Final_time and average test loss
    annotation_text = f"Final Time: {Final_time}\nAvg. Test Error: {test_l2_avg:.4f}"
    plt.annotate(annotation_text, xy=(0.6, 0.8), xycoords="axes fraction",
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Trend problem_{problem} model_{network_name}")

    plot_name = os.path.join(folder, "LossTrend.png")
    plt.savefig(plot_name, dpi=600, bbox_inches='tight')

    # Optional: avoid plt.show() in environments with display issues
    try:
        plt.show()
    except AttributeError:
        print(f"Error showing plot. The loss trend is saved at {plot_name}.")
'''
def plot_loss_trend(losses, labels, problem, network_name, Final_time, test_l2_avg, plot_dir, pde_weight):
    folder = plot_dir
    os.makedirs(folder, exist_ok=True)

    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font

    plt.figure(figsize=(3.5, 4))  # Consider a slightly wider figure if labels/annotations overlap
    for loss_data, label in zip(losses, labels):
        # Ensure loss_data is a list/array of numbers, not a single tensor
        # If loss_data is a PyTorch tensor (e.g., from concatenating loss history)
        if torch.is_tensor(loss_data):
            loss_data_cpu = loss_data.cpu().numpy()  # Move to CPU and convert to NumPy array
        elif isinstance(loss_data, list) and len(loss_data) > 0 and torch.is_tensor(loss_data[0]):
            # If it's a list of tensors (e.g. if each epoch's loss was a tensor)
            # This case is less likely if train_mse_log etc. store scalar Python numbers
            loss_data_cpu = [item.cpu().item() if torch.is_tensor(item) else item for item in loss_data]
        elif isinstance(loss_data, list):
            # If it's already a list of Python numbers (floats/ints)
            loss_data_cpu = loss_data
        else:
            # Fallback or error handling if the format is unexpected
            print(
                f"Warning: Unexpected type for loss data with label '{label}': {type(loss_data)}. Trying to plot directly.")
            loss_data_cpu = loss_data  # Try plotting as is, might fail if it's a GPU tensor

        plt.semilogy(loss_data_cpu, label=label)  # Use the CPU version for plotting

    plt.legend()

    annotation_text = f"Final Time: {Final_time}s\nAvg. Test Error: {test_l2_avg:.4e}"  # Added 's' for seconds, .4e for scientific notation
    # Adjust xy for better placement if needed, or use transform=plt.gca().transAxes
    plt.annotate(annotation_text, xy=(0.55, 0.75), xycoords="axes fraction",
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")  # Clarify log scale
    plt.title(f"Loss Trend: {problem} ({network_name})", fontsize=10)  # Shorter title

    plot_name = os.path.join(folder, f"LossTrend_pde_w{pde_weight}.png")  # Consistent naming
    plt.savefig(plot_name, dpi=600, bbox_inches='tight')
    print(f"Loss trend plot saved to {plot_name}")

    # Optional: avoid plt.show() in environments with display issues
    # try:
    #     plt.show()
    # except Exception as e: # Catch generic Exception for display issues
    #     print(f"Could not display plot due to: {e}. Plot saved at {plot_name}.")
    plt.close(plt.gcf())  # Close the figure to free memory, especially in loops


def plot_field_trajectory(domain, fields, field_names, time_steps, plot_range, problem, network_name, plot_show=True,
                          interpolation=True):
    colors = ["black", "yellow"] if fields[0].ndim == 3 else ["white", "blue"]
    custom_cmap = LinearSegmentedColormap.from_list("two_phase", colors, N=100)
    #folder = problem + "/plots/"
    folder = os.path.join(problem, f"plots_{network_name}") + "/"
    os.makedirs(folder, exist_ok=True)
    interpolation_opt = 'lanczos' if interpolation else 'nearest'

    for time_step in time_steps:
        v_min, v_max = None, None
        for field, field_name, domain_range in zip(fields, field_names, plot_range):
            shot = field[..., time_step]
            if shot.ndim == 3:
                Nx = shot.shape[0]
                Ny = shot.shape[1]
                Nz = shot.shape[2]

                Lx = Ly = Lz = (domain[1] - domain[0])
                hx, hy, hz = Lx / Nx, Ly / Ny, Lz / Nz

                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
                norm = Normalize(vmin=domain_range[0], vmax=domain_range[1])
                sm = ScalarMappable(cmap=custom_cmap, norm=norm)
                sm.set_array([])

                # cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
                # cbar.set_label("Scalar Field Value", fontsize=12)
                verts, faces, _, values = measure.marching_cubes(shot.numpy(), level=np.mean([shot.min(), shot.max()]),
                                                                 spacing=(hx, hy, hz), allow_degenerate=False)
                face_colors = custom_cmap(norm(values))
                p1 = Poly3DCollection(verts[faces], alpha=0.2, facecolors=face_colors)

                p1.set_edgecolor('navy')
                p1.set_facecolor(face_colors)
                p1.set_alpha(0.2)

                ax.add_collection3d(p1)
                ax.set_title(f'{field_name} at T={time_step + 1}')
                ax.set_box_aspect([1, 1, 1])
                # zoom_factor = 0.5
                # ax.set_xlim([-zoom_factor, zoom_factor])
                # ax.set_ylim([-zoom_factor, zoom_factor])
                # ax.set_zlim([-zoom_factor, zoom_factor])
                ax.view_init(elev=35, azim=45)
                ax.set_box_aspect([Nx, Ny, Nz])

                ax.grid(False)
                ax.axis('off')
                # plt.pause(2)

            else:
                plt.figure()
                # plt.imshow(shot, extent=(domain[0], domain[1], domain[0], domain[1]), origin='lower', cmap=custom_cmap,
                #           vmin=domain_range[0], vmax=domain_range[1], aspect='equal', interpolation=interpolation_opt)

                # plt.imshow(shot, extent=(domain[0], domain[1], domain[0], domain[1]), aspect='equal', cmap='jet',
                #            vmin=domain_range[0], vmax=domain_range[1], interpolation=interpolation_opt)

                if v_min is None:
                    v_min = shot.min()
                    v_max = shot.max()

                if field_name == 'Error':
                    v_min = v_min * 0.25
                    v_max = v_max * 0.25

                plt.imshow(shot, extent=(domain[0], domain[1], domain[0], domain[1]), aspect='equal', cmap='jet',
                           vmin=v_min, vmax=v_max, interpolation=interpolation_opt)

                plt.colorbar()
                plt.axis('off')
                # plt.title(f'{field_name} at T={time_step+1}')
            time_step_formatted = str(time_step + 1).zfill(3)
            plot_name = folder + f'{field_name}_at_T_{time_step_formatted}'
            plt.savefig(plot_name + '.png', dpi=300, bbox_inches='tight')
            if plot_show:
                plt.show()
            plt.close()


def make_video(pred, domain, video_name, plot_range, problem, transition_frames=10):
    output_dir = os.path.join(problem, 'video_' + video_name)
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'plots')
    os.makedirs(frames_dir, exist_ok=True)

    time_steps = list(range(pred.shape[-1]))
    fields = [pred]
    field_names = [video_name] * len(time_steps)
    plot_field_trajectory(domain, fields, field_names, time_steps, plot_range, output_dir, False)

    video_path = os.path.join(output_dir, video_name + ".mp4")
    frame_rate = 24
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    for i in range(len(image_files) - 1):
        frame1 = cv2.imread(image_files[i])
        frame2 = cv2.imread(image_files[i + 1])
        video.write(frame1)
        for alpha in np.linspace(0, 1, transition_frames):
            blended_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            video.write(blended_frame)
    video.write(cv2.imread(image_files[-1]))
    video.release()

    print(f"Video saved at {video_path}")


def save_vtk(filename, array, grid_shape):
    # Create a VTK Image Data object to store the 4D array as a time-varying dataset
    grid = vtk.vtkMultiBlockDataSet()

    # Iterate over each time step (Nt) and add the corresponding 3D grid to the VTK object
    for t in range(grid_shape[3]):
        # Extract the 3D slice for the t-th time step
        time_slice = array[..., t]

        # Create a structured grid for the time slice
        time_grid = vtk.vtkStructuredPoints()
        time_grid.SetDimensions(grid_shape[0], grid_shape[1], grid_shape[2])

        # Convert the 3D array to vtk format
        vtk_array = numpy_support.numpy_to_vtk(time_slice.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

        # Add the 3D data to the grid
        time_grid.GetPointData().SetScalars(vtk_array)

        # Add the time grid to the multi-block dataset
        grid.SetBlock(t, time_grid)

    # Create a writer for the multi-block dataset
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

    print(f"Saved VTK file: {filename}")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

''''
def plot_xy_plane_subplots(domain, field, field_name, time_steps, plot_range, problem, network_name, figsize=(12, 8)):
    """
    Plot 2D x-y plane (z=0) subplots of 3D field data at different time steps in a 2x3 grid.
    """
    folder = os.path.join(problem, f"plots_{network_name}")
    os.makedirs(folder, exist_ok=True)

    field_np = field.cpu().numpy() if torch.is_tensor(field) else field

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    vmin, vmax = plot_range
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, time_step in enumerate(time_steps):
        ax = axes[i]
        xy_plane = field_np[:, :, 0, time_step]

        im = ax.imshow(xy_plane,
                       extent=(domain[0], domain[1], domain[0], domain[1]),
                       aspect='equal',
                       cmap='jet',
                       norm=norm,
                       interpolation='bilinear')

        ax.set_title(f't={time_step}', pad=10)
        ax.axis('off')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    fig.suptitle(f'{field_name} Evolution (x-y plane at z=0)', y=1.02, fontsize=14)
    plt.tight_layout()

    # Corrected saving parameters
    plot_name = os.path.join(folder, f'{field_name}_subplots.png')
    plt.savefig(plot_name,
                dpi=300,
                bbox_inches='tight')
    plt.close()
'''


# This function is now fully parametric.
def plot_xy_plane_subplots(domain, field, field_name, time_steps, desired_times, plot_range, problem, network_name):
    """
    Plot 2D x-y plane (z=0) subplots of 3D field data at different time steps.
    The grid size is determined dynamically based on the number of time steps.

    Args:
        ...
        time_steps (list): List of array INDICES to slice the data for plotting.
        desired_times (list): List of the original SIMULATION TIMES for correct labeling.
        ...
    """
    folder = os.path.join(problem, f"plots_{network_name}")
    os.makedirs(folder, exist_ok=True)

    # --- Safety Check: Ensure there are plots to make ---
    if not time_steps:
        print(f"Warning: No valid time steps for '{field_name}'. Skipping plot.")
        return

    field_np = field.cpu().numpy() if torch.is_tensor(field) else field

    # --- Dynamically determine grid size --- ### NEW ###
    n_plots = len(time_steps)
    # Calculate a nice-looking grid layout (e.g., for 7 plots, it will be 3x3)
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / float(cols)))

    # Adjust figsize for better readability based on grid size
    figsize = (cols * 4, rows * 3.5)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

    vmin, vmax = plot_range
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = None  # Initialize im to hold the last image mappable for the colorbar

    # --- Main plotting loop with correct labels and indices --- ### MODIFIED ###
    for i, (time_idx, time_label) in enumerate(zip(time_steps, desired_times)):
        ax = axes[i]

        # Slicing assumes shape is (sx, sy, sz, time). If 5D, use [:, :, 0, time_idx, 0]
        xy_plane = field_np[:, :, 0, time_idx]

        im = ax.imshow(xy_plane,
                       extent=(domain[0], domain[1], domain[0], domain[1]),
                       aspect='equal',
                       cmap='jet',  # or 'viridis'
                       norm=norm,
                       interpolation='bilinear', origin='lower')

        ax.set_title(f't={time_label}', pad=5, fontsize=10)
        ax.axis('off')

    # --- Hide any unused subplots --- ### NEW ###
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    # --- Add a single, shared colorbar to the figure --- ### NEW ###
    # This is much cleaner than one per subplot
    fig.subplots_adjust(right=0.9)  # Make room for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position: [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(f'{field_name} Evolution (x-y plane at z=0)', y=0.98, fontsize=14, fontweight='bold')
    # plt.tight_layout() # tight_layout can interfere with manual colorbar placement

    plot_name = os.path.join(folder, f'{field_name.replace(" ", "_")}_xy_subplots.png')
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    print(f"XY-plane subplot for '{field_name}' saved to {plot_name}")
    plt.close(fig)


'''''
def plot_combined_results_3d(domain, u_exact, u_pred, error, plot_ranges, problem, network_name, plot_dir, pde_weight, time_steps_indices):
    folder = plot_dir
    os.makedirs(folder, exist_ok=True)

    exact_np = u_exact.cpu().numpy() if torch.is_tensor(u_exact) else u_exact
    print(f"Shape of exact_np: {exact_np.shape}")
    pred_np = u_pred.cpu().numpy() if torch.is_tensor(u_pred) else u_pred
    error_np = error.cpu().numpy() if torch.is_tensor(error) else error

    # exact_np, pred_np, error_np shapes: (sx, sy, sz, DT_out, 1_channel)
    # e.g., (32, 32, 32, 20, 1)

    #time_steps_indices = [0, 10, 19]  # Indices for the DT_out dimension
    # Ensure these indices are valid for your DT_out length (e.g., if DT_out=20, max index is 19)

    # time_labels are just for display, can be different from indices
    time_labels = [f't={idx}' for idx in time_steps_indices]  # Or specific simulation times if you know them
    if time_steps_indices[-1] == 19: time_labels[-1] = '20'

    fig = plt.figure(figsize=(12, 9)) # (10, 8)
    axes = []
    for i in range(9):
        col = i % 3
        row = i // 3
        left = 0.05 + col * 0.3
        bottom = 0.7 - row * 0.3  # Adjusted bottom to spread rows more
        ax = fig.add_axes([left, bottom, 0.28, 0.28], projection='3d')
        axes.append(ax)

    fig.suptitle(f'Phase Field Evolution: {problem} ({network_name})', y=1.05, fontsize=14,
                 fontweight='bold')  # Adjusted suptitle y

    phase_cmap = plt.cm.RdBu
    error_cmap = plt.cm.RdYlGn_r  # Or plt.cm.coolwarm for error

    # --- plot_phase_field and plot_error_scatter remain the same ---
    def plot_phase_field(ax, volume_3d, vrange):  # volume_3d should be (sx, sy, sz)
        """Plot phase field with isosurfaces"""
        # Ensure volume_3d is indeed 3D
        if volume_3d.ndim != 3:
            raise ValueError(f"plot_phase_field expects a 3D volume, got shape {volume_3d.shape}")

        spacing = (domain[1] / volume_3d.shape[0],) * 3  # Assuming domain is [L, L] and Lx=Ly=Lz
        level = np.mean(vrange) if vrange else np.mean(volume_3d)  # Ensure vrange is not None

        try:
            # Ensure measure is imported: from skimage import measure
            from skimage import measure
            verts, faces, _, values = measure.marching_cubes(
                volume_3d, level, spacing=spacing, step_size=2)  # Ensure volume is scalar field

            from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Import if not already
            norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1]) if vrange else plt.Normalize(vmin=volume_3d.min(),
                                                                                              vmax=volume_3d.max())
            mesh = Poly3DCollection(verts[faces], alpha=0.7)  # Increased alpha slightly
            # mesh.set_facecolor(phase_cmap(norm(values))) # This can cause issues if values don't align perfectly with verts/faces
            mesh.set_cmap(phase_cmap)  # Apply colormap to the mesh
            mesh.set_array(values)  # Set values for coloring
            mesh.set_clim(norm.vmin, norm.vmax)  # Set color limits
            ax.add_collection3d(mesh)
        except Exception as e:
            print(f"Error in marching_cubes/plotting: {e}")
            pass

        ax.set_xlim(0, domain[1])  # Assuming domain starts at 0 for marching cubes output
        ax.set_ylim(0, domain[1])
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1])  # Enforce cubic aspect ratio
        ax.view_init(elev=30, azim=45)
        ax.set_axis_off()

    def plot_error_scatter(ax, error_vol_3d, vrange):  # error_vol_3d should be (sx, sy, sz)
        """Plot error as colored scatter points"""
        if error_vol_3d.ndim != 3:
            raise ValueError(f"plot_error_scatter expects a 3D volume, got shape {error_vol_3d.shape}")

        sample_step = max(1, error_vol_3d.shape[0] // 15)  # Adjusted sample step
        x_idx, y_idx, z_idx = np.mgrid[0:error_vol_3d.shape[0]:sample_step,
                              0:error_vol_3d.shape[1]:sample_step,
                              0:error_vol_3d.shape[2]:sample_step]
        points = error_vol_3d[x_idx, y_idx, z_idx].flatten()

        # Only plot significant error points to avoid clutter
        threshold = np.percentile(np.abs(points), 60)  # Show more points
        mask = np.abs(points) > threshold
        x_idx, y_idx, z_idx, points_masked = x_idx.flatten()[mask], y_idx.flatten()[mask], z_idx.flatten()[mask], \
        points[mask]

        # Convert indices to spatial coordinates if needed (assuming domain starts at 0)
        # This depends on how 'domain' is defined and used by marching_cubes.
        # If domain is [L_total], then scale by L_total / num_points
        scale_x = domain[1] / error_vol_3d.shape[0]  # Assuming domain[0] is min, domain[1] is max or length
        scale_y = domain[1] / error_vol_3d.shape[1]
        scale_z = domain[1] / error_vol_3d.shape[2]

        x_coord, y_coord, z_coord = x_idx * scale_x, y_idx * scale_y, z_idx * scale_z

        scatter_plot = ax.scatter(x_coord, y_coord, z_coord, c=points_masked,
                                  cmap=error_cmap,
                                  vmin=vrange[0], vmax=vrange[1],
                                  alpha=0.7, s=15, marker='o')  # Increased alpha/size

        ax.set_xlim(0, domain[1])
        ax.set_ylim(0, domain[1])
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=45)
        ax.set_axis_off()

    # Plot exact solutions
    for col_idx, time_idx in enumerate(time_steps_indices):
        ax = axes[col_idx]
        # Slice the time dimension (axis 3) and squeeze the channel dimension (axis 4, which is index -1)
        #volume_to_plot = exact_np[:, :, :, time_idx, 0]  # Shape: (sx, sy, sz)
        volume_to_plot = exact_np[:, :, :, time_idx]  # Shape: (sx, sy, sz)
        plot_phase_field(ax, volume_to_plot, plot_ranges[0])
        ax.set_title(f'Exact, {time_labels[col_idx]}', pad=5, fontsize=11)

    # Plot predicted solutions
    for col_idx, time_idx in enumerate(time_steps_indices):
        ax = axes[col_idx + 3]
        #volume_to_plot = pred_np[:, :, :, time_idx, 0]  # Shape: (sx, sy, sz)
        volume_to_plot = pred_np[:, :, :, time_idx]  # Shape: (sx, sy, sz)
        plot_phase_field(ax, volume_to_plot, plot_ranges[1])
        ax.set_title(f'Predicted, {time_labels[col_idx]}', pad=5, fontsize=11)

    # Plot errors
    for col_idx, time_idx in enumerate(time_steps_indices):
        ax = axes[col_idx + 6]
        #volume_to_plot = error_np[:, :, :, time_idx, 0]  # Shape: (sx, sy, sz)
        volume_to_plot = error_np[:, :, :, time_idx]  # Shape: (sx, sy, sz)
        plot_error_scatter(ax, volume_to_plot, plot_ranges[2])
        ax.set_title(f'Error, {time_labels[col_idx]}', pad=5, fontsize=11)

    # Adjust layout to prevent overlap (might need fine-tuning)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect leaves space for suptitle

    plt.savefig(os.path.join(folder, f'3d_phase_evolution_pde_w{pde_weight}.png'),
                dpi=300, bbox_inches='tight')  # Increased DPI
    print(f"3D phase evolution plot saved to {os.path.join(folder, f'3d_phase_evolution_pde_w{pde_weight}.png')}")
    plt.close(fig)  # Close the figure object
'''

'''
# This function is now fully parametric and will create as many columns as needed.
def plot_combined_results_3d(domain, u_exact, u_pred, error, plot_ranges, problem, network_name, plot_dir, pde_weight,
                             time_steps_indices, desired_times):
    """
    Plot combined 3D results in a dynamic grid (rows: Exact, Predicted, Error; columns: based on time steps).
    This version is parameterized to handle any number of time steps.

    Args:
        ...
        time_steps_indices (list): List of array INDICES to slice the data for plotting.
        desired_times (list): List of the original SIMULATION TIMES for correct labeling.
    """
    folder = plot_dir
    os.makedirs(folder, exist_ok=True)

    # --- Safety Check: Ensure there are plots to make ---
    if not time_steps_indices:
        print("Warning: No valid time steps provided for 3D plot. Skipping.")
        return

    # --- Data Preparation ---
    exact_np = u_exact.cpu().numpy() if torch.is_tensor(u_exact) else u_exact
    pred_np = u_pred.cpu().numpy() if torch.is_tensor(u_pred) else u_pred
    error_np = error.cpu().numpy() if torch.is_tensor(error) else error

    # Assuming input tensors are (sx, sy, sz, T_out). If they are 5D with a channel,
    # the slicing below `[:, :, :, time_idx]` might need to be `[:, :, :, time_idx, 0]`
    print(f"Shape of 3D data for plotting (exact): {exact_np.shape}")

    # --- Dynamic Grid Creation ---
    num_cols = len(time_steps_indices)
    # Adjust figsize based on the number of columns to keep plots from being too squished
    fig_width = 5 * num_cols
    fig_height = 12
    # Create a 3xN grid of 3D subplots. squeeze=False ensures `axes` is always a 2D array.
    fig, axes = plt.subplots(3, num_cols, figsize=(fig_width, fig_height),
                             subplot_kw={'projection': '3d'}, squeeze=False)

    fig.suptitle(f'3D Phase Field Evolution: {problem} ({network_name})', y=0.98, fontsize=16, fontweight='bold')

    # --- Helper plotting functions (kept inside for encapsulation) ---
    def plot_phase_field(ax, volume_3d, vrange, cmap):
        if volume_3d.ndim != 3:
            raise ValueError(f"plot_phase_field expects a 3D volume, got shape {volume_3d.shape}")

        spacing = (domain[1] / volume_3d.shape[0],) * 3
        level = np.mean(vrange)
        verts, faces, _, values = measure.marching_cubes(volume_3d, level, spacing=spacing)

        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
        mesh.set_facecolor(cmap(norm(values)))  # Color faces directly
        mesh.set_edgecolor('k')
        mesh.set_linewidth(0.1)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, domain[1]);
        ax.set_ylim(0, domain[1]);
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1]);
        ax.view_init(elev=30, azim=45);
        ax.set_axis_off()

    def plot_error_scatter(ax, error_vol_3d, vrange, cmap):
        if error_vol_3d.ndim != 3:
            raise ValueError(f"plot_error_scatter expects a 3D volume, got shape {error_vol_3d.shape}")

        sample_step = max(1, error_vol_3d.shape[0] // 15)
        x_idx, y_idx, z_idx = np.mgrid[0:error_vol_3d.shape[0]:sample_step, 0:error_vol_3d.shape[1]:sample_step,
                              0:error_vol_3d.shape[2]:sample_step]
        points = error_vol_3d[x_idx, y_idx, z_idx].flatten()

        threshold = np.percentile(np.abs(points), 60)
        mask = np.abs(points) > threshold
        x, y, z, pts_masked = x_idx.flatten()[mask], y_idx.flatten()[mask], z_idx.flatten()[mask], points[mask]

        scale = domain[1] / error_vol_3d.shape[0]
        ax.scatter(x * scale, y * scale, z * scale, c=pts_masked, cmap=cmap, vmin=vrange[0], vmax=vrange[1], alpha=0.7,
                   s=15)

        ax.set_xlim(0, domain[1]);
        ax.set_ylim(0, domain[1]);
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1]);
        ax.view_init(elev=30, azim=45);
        ax.set_axis_off()

    # --- Main Plotting Loop ---
    # Loop through each time step (column) and plot all three fields (rows)
    for col, (time_idx, time_label) in enumerate(zip(time_steps_indices, desired_times)):
        # Row 0: Exact Solution
        ax_exact = axes[0, col]
        vol_exact = exact_np[:, :, :, time_idx]
        plot_phase_field(ax_exact, vol_exact, plot_ranges[0], plt.cm.viridis)
        ax_exact.set_title(f'Exact, t={time_label}', fontsize=12)

        # Row 1: Predicted Solution
        ax_pred = axes[1, col]
        vol_pred = pred_np[:, :, :, time_idx]
        plot_phase_field(ax_pred, vol_pred, plot_ranges[1], plt.cm.viridis)
        ax_pred.set_title(f'Predicted, t={time_label}', fontsize=12)

        # Row 2: Error
        ax_error = axes[2, col]
        vol_error = error_np[:, :, :, time_idx]
        plot_error_scatter(ax_error, vol_error, plot_ranges[2], plt.cm.coolwarm)
        ax_error.set_title(f'Error, t={time_label}', fontsize=12)

    # Use tight_layout to adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # rect leaves space for suptitle

    # --- Save Figure ---
    plot_filename = os.path.join(folder, f'3d_phase_evolution_pde_w{pde_weight}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"3D phase evolution plot saved to {plot_filename}")
    plt.close(fig)
'''

'''''
def plot_combined_results_3d(domain, u_exact, u_pred, error, plot_ranges, problem, network_name, plot_dir, pde_weight,
                             time_steps_indices, desired_times):
    """
    Plot combined 3D results using the manual layout from the original code,
    but dynamically adapted to the number of time steps provided.
    """
    folder = plot_dir
    os.makedirs(folder, exist_ok=True)

    if not time_steps_indices:
        print("Warning: No valid time steps provided for 3D plot. Skipping.")
        return

    exact_np = u_exact.cpu().numpy() if torch.is_tensor(u_exact) else u_exact
    pred_np = u_pred.cpu().numpy() if torch.is_tensor(u_pred) else u_pred
    error_np = error.cpu().numpy() if torch.is_tensor(error) else error

    print(f"Shape of 3D data for plotting (exact): {exact_np.shape}")

    # --- DYNAMIC MANUAL LAYOUT (mimics old style) ---
    num_cols = len(time_steps_indices)
    fig_width = 4 * num_cols  # Adjust figure width for readability
    fig_height = 9
    fig = plt.figure(figsize=(fig_width, fig_height))

    axes = []
    # Create 3 rows of plots with a dynamic number of columns
    for row in range(3):
        for col in range(num_cols):
            # Calculate positions similar to the old manual style
            ax_width = 1.0 / (num_cols + 1)  # Give some padding
            ax_height = 0.28
            left = 0.05 + col * (ax_width + 0.02)
            bottom = 0.7 - row * (ax_height + 0.05)
            ax = fig.add_axes([left, bottom, ax_width, ax_height], projection='3d')
            axes.append(ax)

    fig.suptitle(f'Phase Field Evolution: {problem} ({network_name})', y=1.0, fontsize=14, fontweight='bold')

    # Use colormaps from the old code for visual consistency
    phase_cmap = plt.cm.RdBu
    error_cmap = plt.cm.RdYlGn_r

    # --- Helper functions from the old code (with minor improvements) ---
    def plot_phase_field(ax, volume_3d, vrange):
        if volume_3d.ndim != 3:
            raise ValueError(f"plot_phase_field expects a 3D volume, got shape {volume_3d.shape}")
        spacing = (domain[1] / volume_3d.shape[0],) * 3
        level = np.mean(vrange) if vrange else np.mean(volume_3d)
        verts, faces, _, values = measure.marching_cubes(volume_3d, level, spacing=spacing, step_size=2)

        norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        mesh.set_cmap(phase_cmap)
        mesh.set_array(values)  # Set values for coloring
        mesh.set_clim(norm.vmin, norm.vmax)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, domain[1]);
        ax.set_ylim(0, domain[1]);
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1]);
        ax.view_init(elev=30, azim=45);
        ax.set_axis_off()

    def plot_error_scatter(ax, error_vol_3d, vrange):
        if error_vol_3d.ndim != 3:
            raise ValueError(f"plot_error_scatter expects a 3D volume, got shape {error_vol_3d.shape}")
        sample_step = max(1, error_vol_3d.shape[0] // 15)
        x_idx, y_idx, z_idx = np.mgrid[0:error_vol_3d.shape[0]:sample_step, 0:error_vol_3d.shape[1]:sample_step,
                              0:error_vol_3d.shape[2]:sample_step]
        points = error_vol_3d[x_idx, y_idx, z_idx].flatten()
        threshold = np.percentile(np.abs(points), 60)
        mask = np.abs(points) > threshold
        x, y, z, pts_masked = x_idx.flatten()[mask], y_idx.flatten()[mask], z_idx.flatten()[mask], points[mask]
        scale = domain[1] / error_vol_3d.shape[0]
        ax.scatter(x * scale, y * scale, z * scale, c=pts_masked, cmap=error_cmap, vmin=vrange[0], vmax=vrange[1],
                   alpha=0.7, s=15)
        ax.set_xlim(0, domain[1]);
        ax.set_ylim(0, domain[1]);
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1]);
        ax.view_init(elev=30, azim=45);
        ax.set_axis_off()

    # --- Plotting loop adapted for the new dynamic axes list ---
    for col_idx, (time_idx, time_label) in enumerate(zip(time_steps_indices, desired_times)):
        # Plot exact solutions (top row)
        ax_exact = axes[col_idx]
        volume_to_plot = exact_np[:, :, :, time_idx]
        plot_phase_field(ax_exact, volume_to_plot, plot_ranges[0])
        ax_exact.set_title(f'Exact, t={time_label}', pad=5, fontsize=11)

        # Plot predicted solutions (middle row)
        ax_pred = axes[col_idx + num_cols]
        volume_to_plot = pred_np[:, :, :, time_idx]
        plot_phase_field(ax_pred, volume_to_plot, plot_ranges[1])
        ax_pred.set_title(f'Predicted, t={time_label}', pad=5, fontsize=11)

        # Plot errors (bottom row)
        ax_error = axes[col_idx + 2 * num_cols]
        volume_to_plot = error_np[:, :, :, time_idx]
        plot_error_scatter(ax_error, volume_to_plot, plot_ranges[2])
        ax_error.set_title(f'Error, t={time_label}', pad=5, fontsize=11)

    # Use tight_layout to adjust spacing, or remove if manual spacing is perfect
    # plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_filename = os.path.join(folder, f'3d_phase_evolution_pde_w{pde_weight}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"3D phase evolution plot saved to {plot_filename}")
    plt.close(fig)
'''


def plot_combined_results_3d(domain, u_exact, u_pred, error, plot_ranges, problem, network_name, plot_dir, pde_weight,
                             time_steps_indices, desired_times):
    """
    Plots combined 3D results.
    This version is corrected to match the reference image by:
    1. Removing the error thresholding to plot ALL sampled points. This creates the
       'yellow cube' for near-zero error and fixes the color scale for other plots.
    2. Keeping the solid color for the phase field plots.
    3. Using a dynamic, symmetric color range for the error plots.
    """
    folder = plot_dir
    os.makedirs(folder, exist_ok=True)

    if not time_steps_indices or len(time_steps_indices) == 0:
        print("Warning: No valid time steps provided for 3D plot. Skipping.")
        return

    exact_np = u_exact.cpu().numpy() if torch.is_tensor(u_exact) else u_exact
    pred_np = u_pred.cpu().numpy() if torch.is_tensor(u_pred) else u_pred
    error_np = error.cpu().numpy() if torch.is_tensor(error) else error

    fig = plt.figure(figsize=(10, 8))
    axes = []
    for i in range(9):
        col = i % 3
        row = i // 3
        left = 0.05 + col * 0.3
        bottom = 0.7 - row * 0.3
        ax = fig.add_axes([left, bottom, 0.28, 0.28], projection='3d')
        axes.append(ax)

    fig.suptitle(f'Phase Field Evolution: {problem} ({network_name})', y=0.98, fontsize=14,
                 fontweight='bold')

    # Calculate a dynamic and symmetric color range for all error plots
    error_slices_to_plot = error_np[:, :, :, time_steps_indices]
    max_abs_error = np.max(np.abs(error_slices_to_plot))
    if max_abs_error < 1e-9:
        max_abs_error = 1e-9
    error_vrange = [-max_abs_error, max_abs_error]

    # --- Corrected Helper Functions ---

    # Helper for phase field (unchanged from last version)
    def plot_phase_field(ax, volume_3d, vrange):
        if volume_3d.ndim != 3:
            raise ValueError(f"plot_phase_field expects a 3D volume, got shape {volume_3d.shape}")

        spacing = (domain[1] / volume_3d.shape[0],) * 3
        level = np.mean(vrange) if vrange else np.mean(volume_3d)

        try:
            verts, faces, _, _ = measure.marching_cubes(
                volume_3d, level, spacing=spacing, step_size=2)
            mesh = Poly3DCollection(verts[faces], alpha=0.7)
            mesh.set_facecolor('darkblue')
            ax.add_collection3d(mesh)
        except Exception as e:
            pass

        ax.set_xlim(0, domain[1]);
        ax.set_ylim(0, domain[1]);
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1]);
        ax.view_init(elev=30, azim=45);
        ax.set_axis_off()

    # **MODIFIED** helper for error plotting
    def plot_error_scatter(ax, error_vol_3d, vrange):
        if error_vol_3d.ndim != 3:
            raise ValueError(f"plot_error_scatter expects a 3D volume, got shape {error_vol_3d.shape}")

        error_cmap = plt.cm.RdYlGn_r
        sample_step = max(1, error_vol_3d.shape[0] // 15)

        # Sample points from the volume
        x_idx, y_idx, z_idx = np.mgrid[0:error_vol_3d.shape[0]:sample_step,
                              0:error_vol_3d.shape[1]:sample_step,
                              0:error_vol_3d.shape[2]:sample_step]

        # Get the error values for the sampled points
        points = error_vol_3d[x_idx, y_idx, z_idx].flatten()

        # --- KEY CHANGE: REMOVED THRESHOLDING ---
        # By removing the threshold, we plot ALL sampled points.
        # This will create the "yellow cube" when error is near zero
        # and correctly use the colormap for other cases.
        x_flat = x_idx.flatten()
        y_flat = y_idx.flatten()
        z_flat = z_idx.flatten()

        # Convert grid indices to spatial coordinates
        scale = domain[1] / error_vol_3d.shape[0]
        x_coord, y_coord, z_coord = x_flat * scale, y_flat * scale, z_flat * scale

        # Plot all the sampled points, using the full 'points' array for color
        ax.scatter(x_coord, y_coord, z_coord, c=points, cmap=error_cmap, vmin=vrange[0], vmax=vrange[1],
                   alpha=0.7, s=15, marker='o')

        ax.set_xlim(0, domain[1]);
        ax.set_ylim(0, domain[1]);
        ax.set_zlim(0, domain[1])
        ax.set_box_aspect([1, 1, 1]);
        ax.view_init(elev=30, azim=45);
        ax.set_axis_off()

    # --- Plotting Loops (unchanged) ---
    for col_idx, time_idx in enumerate(time_steps_indices):
        ax = axes[col_idx]
        volume_to_plot = exact_np[:, :, :, time_idx]
        plot_phase_field(ax, volume_to_plot, plot_ranges[0])
        ax.set_title(f'Exact, t={desired_times[col_idx]}', pad=5, fontsize=9)

    for col_idx, time_idx in enumerate(time_steps_indices):
        ax = axes[col_idx + 3]
        volume_to_plot = pred_np[:, :, :, time_idx]
        plot_phase_field(ax, volume_to_plot, plot_ranges[1])
        ax.set_title(f'Predicted, t={desired_times[col_idx]}', pad=5, fontsize=9)

    for col_idx, time_idx in enumerate(time_steps_indices):
        ax = axes[col_idx + 6]
        volume_to_plot = error_np[:, :, :, time_idx]
        plot_error_scatter(ax, volume_to_plot, error_vrange)
        ax.set_title(f'Error, t={desired_times[col_idx]}', pad=5, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_filename = os.path.join(folder, f'3d_phase_evolution_pde_w{pde_weight}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"3D phase evolution plot saved to {plot_filename}")
    plt.close(fig)

def relative_l2_error_numpy(u_true, u_pred, eps=1e-8):
    """
    Calculates the relative L2 error between two NumPy arrays.
    Assumes u_true and u_pred are for a single time frame and have the same shape.
    """
    diff_norm = np.linalg.norm(u_true.flatten() - u_pred.flatten(), 2)
    true_norm = np.linalg.norm(u_true.flatten(), 2)
    return diff_norm / (true_norm + eps)

'''''
def plot_combined_results(domain, u_exact, u_pred, error, plot_ranges, problem, network_name, plot_dir, pde_weight, time_steps_indices,
                          figsize=(15, 9)):
    """
    Plot combined results in 3x5 grid (rows: Exact, Predicted, Error; columns: time steps)
    """
    folder = plot_dir
    os.makedirs(folder, exist_ok=True)

    # Convert all fields to numpy
    # Assuming u_exact, u_pred, error are already (sx, sy, sz, DT_out, 1_channel) or (sx, sy, DT_out, 1_channel) for 2D spatial
    exact_np = u_exact.cpu().numpy() if torch.is_tensor(u_exact) else u_exact
    pred_np = u_pred.cpu().numpy() if torch.is_tensor(u_pred) else u_pred
    error_np = error.cpu().numpy() if torch.is_tensor(error) else error  # This error is likely u_pred - u_exact

    # Determine if data is 2D spatial or 3D spatial based on ndim
    # exact_np shape e.g., (32, 32, 20, 1) for 2D space + time + channel
    # or (32, 32, 32, 20, 1) for 3D space + time + channel
    is_3d_spatial = exact_np.ndim == 5  # (sx, sy, sz, time, channel)

    # Time steps and labels (showing 20 instead of 19 for the last label)
    # These are indices into the time dimension of your numpy arrays
    #time_steps_indices = [0, 5, 10, 15, 19]  # Ensure these are valid for your time dimension length

    # Create labels based on the indices or actual time values if you have them
    time_labels = [str(idx) for idx in time_steps_indices]  # Default to index
    # If you want the last label to be '20' even if index is 19 (assuming it represents end time)
    if time_steps_indices[-1] == 19: time_labels[-1] = '20'

    fig, axes = plt.subplots(3, 5, figsize=figsize)
    fig.suptitle(f'Evolution: {problem} ({network_name})', y=1.02, fontsize=14, fontweight='bold')

    # Slicing logic based on whether it's 2D or 3D spatial
    # For 2D spatial: exact_np is (sx, sy, time, channel). Slice is exact_np[:, :, t_idx, 0]
    # For 3D spatial: exact_np is (sx, sy, sz, time, channel). Slice is exact_np[:, :, 0, t_idx, 0] (showing z=0 slice)

    # Plot Exact solutions (first row)
    for col, (t_idx, t_label) in enumerate(zip(time_steps_indices, time_labels)):
        ax = axes[0, col]
        if is_3d_spatial:
            slice_to_plot = exact_np[:, :, 0, t_idx, 0]  # Show z=0 slice for 3D data
            title_suffix = " (z=0 slice)"
        else:
            slice_to_plot = exact_np[:, :, t_idx, 0]  # For 2D spatial data
            title_suffix = ""

        im = ax.imshow(slice_to_plot,
                       extent=(domain[0], domain[1], domain[0], domain[1]),  # Assuming domain applies to x,y
                       aspect='auto',  # Changed to auto for flexibility
                       cmap='viridis',  # Changed cmap for better perception
                       vmin=plot_ranges[0][0],
                       vmax=plot_ranges[0][1],
                       interpolation='bilinear', origin='lower')
        ax.set_title(f't={t_label}{title_suffix}', fontsize=10)
        ax.axis('off')
        if col == 0: ax.set_ylabel('Exact', fontsize=12, labelpad=10)

        if col == len(time_steps_indices) - 1:  # Colorbar for the last plot in the row
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)  # Removed label for brevity

    # Plot Predicted solutions (second row)
    for col, (t_idx, t_label) in enumerate(zip(time_steps_indices, time_labels)):
        ax = axes[1, col]
        if is_3d_spatial:
            slice_to_plot = pred_np[:, :, 0, t_idx, 0]
        else:
            slice_to_plot = pred_np[:, :, t_idx, 0]

        im = ax.imshow(slice_to_plot,
                       extent=(domain[0], domain[1], domain[0], domain[1]),
                       aspect='auto', cmap='viridis',
                       vmin=plot_ranges[1][0], vmax=plot_ranges[1][1],
                       interpolation='bilinear', origin='lower')
        ax.set_title(f't={t_label}', fontsize=10)  # Title only time for predicted and error
        ax.axis('off')
        if col == 0: ax.set_ylabel('Predicted', fontsize=12, labelpad=10)

        if col == len(time_steps_indices) - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    # Plot Errors (third row) and calculate/display Rel. L2 Error
    for col, (t_idx, t_label) in enumerate(zip(time_steps_indices, time_labels)):
        ax = axes[2, col]

        # Get the specific time slices for error calculation and plotting
        if is_3d_spatial:
            exact_slice_for_error = exact_np[:, :, :, t_idx, 0]  # Full 3D slice for error calc
            error_slice_to_plot = error_np[:, :, 0, t_idx, 0]  # z=0 slice for plotting error field
            pred_slice_for_error = pred_np[:, :, :, t_idx, 0]  # Full 3D slice for error calc
        else:  # 2D spatial
            exact_slice_for_error = exact_np[:, :, t_idx, 0]
            error_slice_to_plot = error_np[:, :, t_idx, 0]
            pred_slice_for_error = pred_np[:, :, t_idx, 0]

        im = ax.imshow(error_slice_to_plot,
                       extent=(domain[0], domain[1], domain[0], domain[1]),
                       aspect='auto', cmap='coolwarm',  # Changed cmap for error
                       vmin=plot_ranges[2][0], vmax=plot_ranges[2][1],
                       interpolation='bilinear', origin='lower')

        # Calculate and display relative L2 error for this time frame
        rel_l2 = relative_l2_error_numpy(exact_slice_for_error, pred_slice_for_error)
        ax.set_title(f't={t_label}\nL2 Err: {rel_l2:.3e}', fontsize=9)  # Display error in title

        ax.axis('off')
        if col == 0: ax.set_ylabel('Error Field', fontsize=12, labelpad=10)

        if col == len(time_steps_indices) - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    # Remove old row labels as they are now per-plot in the first column
    # for row, label in zip(range(3), ['Exact', 'Predicted', 'Error']):
    #     axes[row, 0].text(-0.08, 0.5, label, ...)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for suptitle
    plot_name = os.path.join(folder, f'combined_results_2d_slices_pde_w{pde_weight}.png')
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    print(f"Combined 2D slice plot saved to {plot_name}")
    plt.close(fig)
'''
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_combined_results(domain, u_exact, u_pred, error, plot_ranges, problem, network_name, plot_dir, pde_weight,
                          time_steps_indices, desired_times):
    """
    Plot combined results in a dynamic grid (rows: Exact, Predicted, Error; columns: based on time steps).
    This version is corrected to properly handle 4D (3D space + time) and 3D (2D space + time) data.
    """
    folder = plot_dir
    os.makedirs(folder, exist_ok=True)

    if not time_steps_indices:
        print("Warning: No valid time steps provided to plot. Skipping combined plot.")
        return

    exact_np = u_exact.cpu().numpy() if torch.is_tensor(u_exact) else u_exact
    pred_np = u_pred.cpu().numpy() if torch.is_tensor(u_pred) else u_pred
    error_np = error.cpu().numpy() if torch.is_tensor(error) else error

    # --- ### FIXED DIMENSION CHECKING ### ---
    # Your data is 4D: (sx, sy, sz, time). So 3D spatial is ndim==4.
    # If data were 3D: (sx, sy, time), then ndim==3.
    is_3d_spatial = (exact_np.ndim == 4)
    if is_3d_spatial:
        print("Plotting logic: Detected 3D spatial data (4D tensor).")
    else:
        print("Plotting logic: Detected 2D spatial data (3D tensor).")

    time_labels = [str(t) for t in desired_times]
    num_cols = len(time_steps_indices)
    fig, axes = plt.subplots(3, num_cols, figsize=(4 * num_cols, 9), squeeze=False)
    fig.suptitle(f'Evolution: {problem} ({network_name}) - PDE Weight: {pde_weight}', y=1.02, fontsize=14,
                 fontweight='bold')

    for col, (t_idx, t_label) in enumerate(zip(time_steps_indices, time_labels)):
        # --- ### FIXED SLICING LOGIC ### ---
        if is_3d_spatial:
            z_slice_index = exact_np.shape[2] // 2  # Take a slice from the middle of the z-axis
            slice_exact = exact_np[:, :, z_slice_index, t_idx]
            slice_pred = pred_np[:, :, z_slice_index, t_idx]
            slice_error = error_np[:, :, z_slice_index, t_idx]
            title_suffix = f" (z-slice at {z_slice_index})"
            # For error calculation, we use the full 3D volume at that time step
            exact_for_error_calc = exact_np[:, :, :, t_idx]
            pred_for_error_calc = pred_np[:, :, :, t_idx]
        else:  # Assumes 2D spatial data (3D tensor)
            slice_exact = exact_np[:, :, t_idx]
            slice_pred = pred_np[:, :, t_idx]
            slice_error = error_np[:, :, t_idx]
            title_suffix = ""
            exact_for_error_calc = slice_exact
            pred_for_error_calc = slice_pred

        # Plot Exact
        ax_exact = axes[0, col]
        im_exact = ax_exact.imshow(slice_exact, extent=(domain[0], domain[1], domain[0], domain[1]), aspect='auto',
                                   cmap='viridis', vmin=plot_ranges[0][0], vmax=plot_ranges[0][1],
                                   interpolation='bilinear', origin='lower')
        ax_exact.set_title(f't={t_label}{title_suffix}', fontsize=10)
        ax_exact.axis('off')
        if col == 0: ax_exact.set_ylabel('Exact', fontsize=12, labelpad=10)

        # Plot Predicted
        ax_pred = axes[1, col]
        im_pred = ax_pred.imshow(slice_pred, extent=(domain[0], domain[1], domain[0], domain[1]), aspect='auto',
                                 cmap='viridis', vmin=plot_ranges[1][0], vmax=plot_ranges[1][1],
                                 interpolation='bilinear', origin='lower')
        ax_pred.set_title(f't={t_label}', fontsize=10)
        ax_pred.axis('off')
        if col == 0: ax_pred.set_ylabel('Predicted', fontsize=12, labelpad=10)

        # Plot Error
        ax_error = axes[2, col]
        im_error = ax_error.imshow(slice_error, extent=(domain[0], domain[1], domain[0], domain[1]), aspect='auto',
                                   cmap='coolwarm', vmin=plot_ranges[2][0], vmax=plot_ranges[2][1],
                                   interpolation='bilinear', origin='lower')
        rel_l2 = relative_l2_error_numpy(exact_for_error_calc, pred_for_error_calc)
        ax_error.set_title(f't={t_label}\nL2 Err: {rel_l2:.3e}', fontsize=9)
        ax_error.axis('off')
        if col == 0: ax_error.set_ylabel('Error Field', fontsize=12, labelpad=10)

    # Add colorbars to the last column
    for row, im in enumerate([im_exact, im_pred, im_error]):
        divider = make_axes_locatable(axes[row, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_name = os.path.join(folder, f'combined_results_2d_slices_pde_w{pde_weight}.png')
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    print(f"Combined 2D slice plot saved to {plot_name}")
    plt.close(fig)