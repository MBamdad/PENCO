import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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


def plot_field_history(domain, fields, field_names, time_steps, problem):
    colors = ["black", "yellow"]
    custom_cmap = LinearSegmentedColormap.from_list("black_yellow", colors, N=2)
    folder = problem + "/plots/"
    s = fields[0].shape[0]
    x_test_plot = np.linspace(domain[0], domain[1], s).astype('float32')
    y_test_plot = np.linspace(domain[0], domain[1], s).astype('float32')

    for time_step in time_steps:
        for field, field_name in zip(fields, field_names):
            shot = field[:, :, time_step]
            plt.figure()
            plt.contourf(x_test_plot, y_test_plot, shot, levels=2, cmap=custom_cmap)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(f'{field_name} at T={time_step}')
            plot_name = folder + f'{field_name} at T_{time_step}'
            plt.savefig(plot_name + '.png', dpi=600, bbox_inches='tight')
            plt.show()


def make_video(pred, domain, problem):
    colors = ["black", "yellow"]
    custom_cmap = LinearSegmentedColormap.from_list("black_yellow", colors, N=2)
    output_dir = problem + '/video'
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    s = pred.shape[0]
    x_test_plot = np.linspace(domain[0], domain[1], s).astype('float32')
    y_test_plot = np.linspace(domain[0], domain[1], s).astype('float32')

    for T_index in range(pred.shape[-1]):
        u_pred = pred[:, :, T_index]
        plt.contourf(x_test_plot, y_test_plot, u_pred, levels=2, cmap=custom_cmap)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Predicted Value at T_index = {T_index}')
        image_path = os.path.join(frames_dir, f"frame_{T_index:04d}.png")
        plt.savefig(image_path, dpi=600, bbox_inches='tight')
        plt.close()

    video_path = os.path.join(output_dir, "output_video.mp4")
    frame_rate = 10  # frames per second

    # Get list of saved image paths
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])

    # Create a video writer object
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    # Write frames to video
    for image_file in image_files:
        video.write(cv2.imread(image_file))

    video.release()
    print(f"Video saved at {video_path}")
