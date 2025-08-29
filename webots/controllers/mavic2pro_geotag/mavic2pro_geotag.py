import cv2
from controller import Robot, Keyboard
from math import pow
import numpy as np
import time
import math
from utils import *
from frame_client import FrameClient


robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

camera_roll_motor = robot.getDevice("camera roll")
camera_pitch_motor = robot.getDevice("camera pitch")
try:
    camera_yaw_motor = robot.getDevice("camera yaw")
except Exception:
    camera_yaw_motor = None

initial_roll, initial_pitch, _ = imu.getRollPitchYaw()

front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

print("[client] Establishing connection to simulated ground station")
sender = FrameClient(host="127.0.0.1", port=5001, timeout=5.0, jpeg_quality=80)

for m in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

print("Start the drone...")

cam_width = camera.getWidth()
cam_height = camera.getHeight()
print(f"Camera device: {camera.getName()} | {cam_width}x{cam_height}")


def camera_frame_bgr():
    buf = camera.getImage()
    if buf is None:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size != cam_width * cam_height * 4:
        return None
    arr = arr.reshape((cam_height, cam_width, 4))
    bgr = arr[:, :, :3].copy()
    return bgr


while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

control_mode = ControlMode.MANUAL
target_altitude = 1.0

waypoints = load_waypoints()
curr_waypoint_idx = 0
waypoints_travel = [Status.UNVISITED for i in range(len(waypoints))]
if waypoints:
    waypoints_travel[curr_waypoint_idx] = Status.APPROACHING

cam_pitch_offset = 0.0
cam_yaw_offset = 0.0
cam_pitch_step = 0.005
cam_yaw_step = 0.005
cam_pitch_limit = 0.5
cam_yaw_limit = 1.6

last_p_key_time = 0.0
p_key_debounce_delay = 0.5

print_manual_controls()

while robot.step(timestep) != -1:
    now = robot.getTime()

    roll, pitch, _yaw = imu.getRollPitchYaw()
    gx, gy, _gz = gyro.getValues()
    _x, _y, altitude = gps.getValues()

    frame_bgr = camera_frame_bgr()
    if frame_bgr is not None:
        pos = gps.getValues()
        t = np.array(pos).reshape(
            3,
        )

        R_cam_yaw = np.array(
            [
                [np.cos(cam_yaw_offset), -np.sin(cam_yaw_offset), 0],
                [np.sin(cam_yaw_offset), np.cos(cam_yaw_offset), 0],
                [0, 0, 1],
            ]
        )

        R_cam_pitch = np.array(
            [
                [np.cos(cam_pitch_offset), 0, np.sin(cam_pitch_offset)],
                [0, 1, 0],
                [-np.sin(cam_pitch_offset), 0, np.cos(cam_pitch_offset)],
            ]
        )

        R_cam = R_cam_yaw @ R_cam_pitch

        roll, pitch, yaw = imu.getRollPitchYaw()
        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        R = Rz @ Ry @ Rx @ R_cam
        t = -1 * R.T @ t
        R = R.T

        fov = camera.getFov()
        width = camera.getWidth()
        height = camera.getHeight()

        K = get_camera_intrinsic(height, width, fov)
        sender.send_frame(frame_bgr, K, R, t)

    led_state = int(now) % 2
    front_left_led.set(led_state)
    front_right_led.set(1 - led_state)

    camera_roll_motor.setPosition(-0.115 * gx)
    camera_pitch_motor.setPosition(-0.1 * gy + cam_pitch_offset)
    if camera_yaw_motor is not None:
        camera_yaw_motor.setPosition(cam_yaw_offset)

    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    key = keyboard.getKey()
    while key != -1:
        if key in (ord("P"), ord("p")):
            if now - last_p_key_time > p_key_debounce_delay:
                last_p_key_time = now
                if control_mode == ControlMode.MANUAL:
                    control_mode = ControlMode.AUTO
                    target_altitude = 20.0
                    if waypoints:
                        curr_waypoint_idx = 0
                        waypoints_travel = [
                            Status.UNVISITED for i in range(len(waypoints))
                        ]
                        waypoints_travel[curr_waypoint_idx] = Status.APPROACHING
                    clear_terminal()
                    print(f"\n*** SWITCHED TO {control_mode.value.upper()} MODE ***")
                    time.sleep(0.1)
                else:
                    control_mode = ControlMode.MANUAL
                    target_altitude = 1.0
                    print_manual_controls()
        elif control_mode == ControlMode.MANUAL:
            if key in (ord("W"), ord("w")):
                pitch_disturbance = -2.0
            elif key in (ord("S"), ord("s")):
                pitch_disturbance = 2.0
            elif key in (ord("A"), ord("a")):
                roll_disturbance = 1.0
            elif key in (ord("D"), ord("d")):
                roll_disturbance = -1.0
            elif key in (ord("Q"), ord("q")):
                yaw_disturbance = 2.0
            elif key in (ord("E"), ord("e")):
                yaw_disturbance = -2.0
            elif key in (ord("R"), ord("r")):
                target_altitude += 0.05
                print(f"target altitude: {target_altitude:.2f} m")
            elif key in (ord("F"), ord("f")):
                target_altitude -= 0.05
                print(f"target altitude: {target_altitude:.2f} m")

        if key == Keyboard.UP:
            cam_pitch_offset = clamp(
                cam_pitch_offset - cam_pitch_step, -cam_pitch_limit, cam_pitch_limit
            )
        elif key == Keyboard.DOWN:
            cam_pitch_offset = clamp(
                cam_pitch_offset + cam_pitch_step, -cam_pitch_limit, cam_pitch_limit
            )
        elif key == Keyboard.LEFT:
            cam_yaw_offset = clamp(
                cam_yaw_offset + cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )
        elif key == Keyboard.RIGHT:
            cam_yaw_offset = clamp(
                cam_yaw_offset - cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )

        key = keyboard.getKey()

    if control_mode == ControlMode.AUTO and waypoints:
        target_altitude = 20.0
        pitch_disturbance = -2.0

        wx, wy = waypoints[curr_waypoint_idx]
        theta_desired = math.atan2(wy - _y, wx - _x)
        yaw_disturbance = wrap_to_pi(theta_desired - _yaw)

        if dist(_x, _y, wx, wy) < 8.5:
            waypoints_travel[curr_waypoint_idx] = Status.VISITED
            curr_waypoint_idx = (curr_waypoint_idx + 1) % len(waypoints)
            waypoints_travel[curr_waypoint_idx] = Status.APPROACHING
            wx, wy = waypoints[curr_waypoint_idx]

        frame_count = int(now * 1000 / timestep)
        print_waypoints_status(
            waypoints, waypoints_travel, _x, _y, frame_count, control_mode
        )

    pitch_input = (
        k_pitch_p * clamp(pitch - initial_pitch, -1.0, 1.0) + gy + pitch_disturbance
    )
    roll_input = (
        k_roll_p * clamp(roll - initial_roll, -1.0, 1.0) + gx + roll_disturbance
    )

    if control_mode == ControlMode.AUTO:
        yaw_input = yaw_disturbance * 0.35
    else:
        yaw_input = yaw_disturbance

    clamped_diff_alt = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * pow(clamped_diff_alt, 3.0)

    front_left_motor_input = (
        k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    )
    front_right_motor_input = (
        k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    )
    rear_left_motor_input = (
        k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    )
    rear_right_motor_input = (
        k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
    )

    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)
