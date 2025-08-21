import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List
import argparse
import cv2
import json


def create_sidebyside_view(animation_path: str, raw_video_path: str, save_path: str):
    cap = cv2.VideoCapture(raw_video_path)
    anime_cap = cv2.VideoCapture(animation_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(save_path, fourcc, fps, (width * 2, height))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        ret, anime_frame = anime_cap.read()
        if not ret:
            break
        anime_frame = cv2.resize(anime_frame, (width, height))
        frame = cv2.resize(frame, (width, height))
        anime_frame = cv2.cvtColor(anime_frame, cv2.COLOR_BGR2RGB)

        sidebyside_frame = np.concatenate((frame, anime_frame), axis=1)
        out.write(sidebyside_frame)

    cap.release()
    anime_cap.release()
    out.release()


def animate_trajectory(
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    axis_len: float = 5,
    animation_save_path: str = None,
    raw_video_path: str = None,
    sidebyside_save_path: str = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    traj = np.array([t.flatten() for t in t_list])
    ax.set_xlim(np.min(traj[:, 0]) - 0.5, np.max(traj[:, 0]) + 0.5)
    ax.set_ylim(np.min(traj[:, 1]) - 0.5, np.max(traj[:, 1]) + 0.5)
    ax.set_zlim(np.min(traj[:, 2]) - 0.5, np.max(traj[:, 2]) + 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Visualized trajectory")

    (line,) = ax.plot([], [], [], "b.-", label="Trajectory")
    quivers = []

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return [line]

    def update(frame):
        line.set_data(traj[: frame + 1, 0], traj[: frame + 1, 1])
        line.set_3d_properties(traj[: frame + 1, 2])

        for q in quivers:
            q.remove()
        quivers.clear()

        R = R_list[frame]
        t = t_list[frame].flatten()
        x_axis = R[:, 0] * axis_len
        y_axis = R[:, 1] * axis_len
        z_axis = R[:, 2] * axis_len

        quivers.append(
            ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color="r")
        )
        quivers.append(
            ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color="g")
        )
        quivers.append(
            ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color="b")
        )

        return [line] + quivers

    ani = FuncAnimation(
        fig, update, frames=len(R_list), init_func=init, blit=False, interval=200
    )

    if animation_save_path:
        ani.save(animation_save_path, writer="ffmpeg", fps=60)
        if raw_video_path:
            create_sidebyside_view(
                animation_save_path, raw_video_path, sidebyside_save_path
            )
    else:
        plt.show()


def load_nav_data(nav_data_path: str):
    R_list = []
    t_list = []
    with open(nav_data_path, "r") as f:
        nav_data = json.load(f)

    for frame in nav_data:
        R_list.append(np.array(frame["data"]["R"]))
        t_list.append(np.array(frame["data"]["t"]))
    return R_list, t_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nav-data-path", type=str, required=True, help="Path to the navigation data"
    )
    parser.add_argument(
        "--raw-video-path", type=str, default=None, help="Path to the raw video"
    )
    parser.add_argument(
        "--animation-save-path",
        type=str,
        default="animation.mp4",
        help="Path to save the animation",
    )
    parser.add_argument(
        "--sidebyside-save-path",
        type=str,
        default="sidebyside.mp4",
        help="Path to save the sidebyside view",
    )
    args = parser.parse_args()

    R_list, t_list = load_nav_data(args.nav_data_path)
    animate_trajectory(
        R_list,
        t_list,
        axis_len=5,
        raw_video_path=args.raw_video_path,
        animation_save_path=args.animation_save_path,
        sidebyside_save_path=args.sidebyside_save_path,
    )
