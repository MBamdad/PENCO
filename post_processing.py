import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.cm import ScalarMappable


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


def plot_field_trajectory(domain, fields, field_names, time_steps, plot_range, problem, plot_show=True, interpolation=True):
    colors = ["black", "yellow"] if fields[0].ndim == 3 else ["white", "blue"]
    custom_cmap = LinearSegmentedColormap.from_list("two_phase", colors, N=100)
    folder = problem + "/plots/"
    os.makedirs(folder, exist_ok=True)
    interpolation_opt = 'lanczos' if interpolation else 'nearest'

    for time_step in time_steps:
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
                #verts, faces, _, values = measure.marching_cubes(shot.numpy(), level=0.8*domain_range[1], spacing=(hx, hy, hz), allow_degenerate=False)
                verts, faces, _, values = measure.marching_cubes(shot.numpy(), level=np.mean(domain_range), spacing=(hx, hy, hz), allow_degenerate=False)
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
                plt.imshow(shot, extent=(domain[0], domain[1], domain[0], domain[1]), origin='lower', cmap=custom_cmap,
                           vmin=domain_range[0], vmax=domain_range[1], aspect='equal', interpolation=interpolation_opt)
                # plt.colorbar()
                plt.axis('off')
                # plt.title(f'{field_name} at T={time_step+1}')
            time_step_formatted = str(time_step+1).zfill(3)
            plot_name = folder + f'{field_name}_at_T_{time_step_formatted}'
            plt.savefig(plot_name + '.png', dpi=300, bbox_inches='tight')
            if plot_show:
                plt.show()
            # plt.close()


def make_video(pred, domain, video_name, plot_range, problem, transition_frames=10):
    output_dir = os.path.join(problem, 'video')
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "plots"+video_name)
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
