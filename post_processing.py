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


def plot_field_history(domain, fields, field_names, time_steps, plot_range , problem, plot_show=True, interpolation=True):
    colors = ["black", "yellow"]
    custom_cmap = LinearSegmentedColormap.from_list("black_yellow", colors, N=100)
    folder = problem + "/plots/"
    os.makedirs(folder, exist_ok=True)
    # s = fields[0].shape[0]
    # x_test_plot = np.linspace(domain[0], domain[1], s).astype('float32')
    # y_test_plot = np.linspace(domain[0], domain[1], s).astype('float32')

    interpolation_opt = 'lanczos' if interpolation else 'nearest'

    for time_step in time_steps:
        for field, field_name, range in zip(fields, field_names, plot_range):
            shot = field[:, :, time_step]
            plt.figure()
            plt.imshow(shot, extent=[domain[0], domain[1], domain[0], domain[1]], origin='lower', cmap=custom_cmap,
                       vmin=range[0], vmax=range[1], aspect='equal', interpolation=interpolation_opt)
            #plt.colorbar()
            plt.axis('off')
            #plt.title(f'{field_name} at T={time_step+1}')
            time_step_formatted = str(time_step+1).zfill(3)
            plot_name = folder + f'{field_name} at T_{time_step_formatted}'
            plt.savefig(plot_name + '.png', dpi=1200, bbox_inches='tight')
            if plot_show:
                plt.show()
            plt.close()


def make_video(pred, domain, video_name, plot_range, problem, transition_frames=10):
    output_dir = os.path.join(problem, 'video')
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "plots")
    os.makedirs(frames_dir, exist_ok=True)

    time_steps = list(range(pred.shape[-1]))
    fields = [pred]
    field_names = [video_name] * len(time_steps)
    plot_field_history(domain, fields, field_names, time_steps, plot_range, output_dir, False)

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
