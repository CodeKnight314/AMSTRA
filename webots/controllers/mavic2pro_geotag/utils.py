import time
import random
import os
import math
import numpy as np
from enum import Enum


class Status(Enum):
    UNVISITED = "Unvisited"
    APPROACHING = "Approaching"
    VISITED = "Visited"


class ControlMode(Enum):
    MANUAL = "Manual"
    AUTO = "Auto"


def transmission_delay(min_ms: int = 15, max_ms: int = 25):
    sleep_duration_ms = random.randint(min_ms, max_ms)
    sleep_duration_sec = sleep_duration_ms / 1000
    time.sleep(sleep_duration_sec)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clear_terminal():
    if os.name == "nt":
        _ = os.system("cls")
    else:
        _ = os.system("clear")


def get_camera_intrinsic(height: float, width: float, fov: float):
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    fx = width / (2 * math.tan(fov / 2))
    fy = fx
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return matrix


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def pos_to_waypoints(current_x, current_y, waypts):
    pts_dist = []
    for i, (pts_x, pts_y) in waypts:
        pts_dist.append(dist(current_x, current_y, pts_x, pts_y))
    return pts_dist


def load_waypoints():
    path = "waypoints.txt"
    waypoints = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                data = line.strip().split(", ")
                try:
                    x = int(data[0])
                    y = int(data[1])
                except ValueError:
                    print(f"Malformed data could not be read for line: {line}")
                    continue
                waypoints.append((x, y))
    else:
        waypoints = [(0, -180), (0, 180), (-180, 0), (180, 0)]
    return waypoints


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def print_manual_controls():
    clear_terminal()
    print("\n" + "=" * 60)
    print("DRONE CONTROLLER - MANUAL MODE")
    print("=" * 60)
    print("Controls:")
    print("- WASD: move (W/S=fwd/back, A/D=strafe)")
    print("- Q/E: yaw left/right")
    print("- R/F: altitude up/down")
    print("- Arrow keys: control camera (Left/Right=yaw, Up/Down=pitch)")
    print("- P: Toggle to AUTO mode")
    print("=" * 60)


def print_waypoints_status(
    waypoints, waypoints_travel, current_x, current_y, frame_count, control_mode
):
    if frame_count % 60 != 0 or control_mode != ControlMode.AUTO:
        return
    clear_terminal()
    print("\n" + "=" * 60)
    print("DRONE CONTROLLER - AUTO MODE")
    print("=" * 60)
    print(f"{'Waypoint':<15} {'Status':<15} {'Distance':<10}")
    print("-" * 50)

    approaching_waypoint = None
    for i, ((wx, wy), status_val) in enumerate(zip(waypoints, waypoints_travel)):
        distance = dist(current_x, current_y, wx, wy)
        print(f"({wx:<6}, {wy:<6}) {status_val.value:<15} {distance:<10.2f}")

        if status_val == Status.APPROACHING:
            approaching_waypoint = (wx, wy)

    print("\nInfo:", end=" ")
    if approaching_waypoint:
        print(f"Drone is approaching Waypoint: {approaching_waypoint}")
    print("- P: Toggle to MANUAL mode")
    print("=" * 60)
