"""
Simplistic drone control in Webots (Python version of the provided C controller)
- Stabilize roll/pitch/yaw with PID-like terms
- Cubic vertical control for altitude hold
- Camera stabilization + manual camera control with arrow keys
- WASD + Q/E/R/F drone control
- Keyboard controls
"""

from controller import Robot, Keyboard
from math import pow
import os
import time

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --- Initialization ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Devices
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
# Try to get camera yaw if available on the robot (optional)
try:
    camera_yaw_motor = robot.getDevice("camera yaw")
except Exception:
    camera_yaw_motor = None

# Propeller motors (velocity control mode)
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

for m in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

print("Start the drone...")

# Camera stream info
cam_width = camera.getWidth()
cam_height = camera.getHeight()
print(f"Camera device: {camera.getName()} | {cam_width}x{cam_height}")


# Helper: get current frame as numpy BGR (safe copy)
def camera_frame_bgr():
    """Return current camera frame as BGR numpy array (H,W,3), or None if unavailable.
    Webots Camera returns BGRA; we drop A and COPY to detach from Webots' buffer (it becomes invalid after next step).
    """
    if not _HAS_NUMPY:
        return None
    buf = camera.getImage()  # raw BGRA bytes from Webots
    if buf is None:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size != cam_width * cam_height * 4:
        return None
    arr = arr.reshape((cam_height, cam_width, 4))
    bgr = arr[:, :, :3].copy()  # BGRA -> BGR, copy() is IMPORTANT
    return bgr


# Optional: lightweight sanity check saver (disabled by default)
_SAVE_DEBUG_JPG = False
_last_save_t = 0.0
_save_interval = 1.0  # seconds

# Wait ~1 second
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

print("Controls:")
print("- WASD: move (W/S=fwd/back, A/D=strafe)")
print("- Q/E: yaw left/right")
print("- R/F: altitude up/down")
print("- Arrow keys: control camera (Left/Right=yaw, Up/Down=pitch)")

# Constants (empirically chosen)
k_vertical_thrust = 68.5  # Thrust at which the drone lifts
k_vertical_offset = 0.6  # Vertical offset where the robot stabilizes
k_vertical_p = 3.0  # P constant for vertical control
k_roll_p = 50.0  # P constant for roll PID
k_pitch_p = 30.0  # P constant for pitch PID

# Variables
target_altitude = 1.0  # meters

# Camera manual offsets (added on top of gyro stabilization)
cam_pitch_offset = 0.0
cam_yaw_offset = 0.0
cam_pitch_step = 0.02  # radians per key press
cam_yaw_step = 0.03  # radians per key press
cam_pitch_limit = 0.6  # clamp range
cam_yaw_limit = 1.6

# --- Main loop ---
while robot.step(timestep) != -1:
    now = robot.getTime()

    # Sensor reads
    roll, pitch, _yaw = imu.getRollPitchYaw()
    gx, gy, _gz = gyro.getValues()
    _x, _y, altitude = gps.getValues()

    # === Camera frame for postprocessing (YOLO/Depth) ===
    # Option A (raw BGRA bytes):
    frame_bytes = (
        camera.getImage()
    )  # <-- YOLO/Depth entrypoint (BGRA). This pointer is only valid UNTIL NEXT robot.step().

    # Option B (preferred for OpenCV/YOLO): safe numpy BGR copy
    frame_bgr = camera_frame_bgr()  # <-- Use this for cv2/YOLO/Depth

    # Blink LEDs at 1 Hz (alternating)
    led_state = int(now) % 2
    front_left_led.set(led_state)
    front_right_led.set(1 - led_state)

    # Stabilize camera using gyro feedback + manual offsets
    camera_roll_motor.setPosition(-0.115 * gx)
    camera_pitch_motor.setPosition(-0.1 * gy + cam_pitch_offset)
    if camera_yaw_motor is not None:
        camera_yaw_motor.setPosition(cam_yaw_offset)

    # Keyboard disturbances
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    key = keyboard.getKey()
    while key != -1:
        # --- Drone motion: WASD/Q/E/R/F ---
        if key in (ord("W"), ord("w")):
            pitch_disturbance = -2.0  # forward
        elif key in (ord("S"), ord("s")):
            pitch_disturbance = 2.0  # backward
        elif key in (ord("A"), ord("a")):
            roll_disturbance = 1.0  # strafe left
        elif key in (ord("D"), ord("d")):
            roll_disturbance = -1.0  # strafe right
        elif key in (ord("Q"), ord("q")):
            yaw_disturbance = 1.3  # yaw left
        elif key in (ord("E"), ord("e")):
            yaw_disturbance = -1.3  # yaw right
        elif key in (ord("R"), ord("r")):
            target_altitude += 0.05
            print(f"target altitude: {target_altitude:.2f} m")
        elif key in (ord("F"), ord("f")):
            target_altitude -= 0.05
            print(f"target altitude: {target_altitude:.2f} m")

        # --- Camera control: Arrow keys ---
        elif key == Keyboard.UP:
            cam_pitch_offset = clamp(
                cam_pitch_offset - cam_pitch_step, -cam_pitch_limit, cam_pitch_limit
            )
        elif key == Keyboard.DOWN:
            cam_pitch_offset = clamp(
                cam_pitch_offset + cam_pitch_step, -cam_pitch_limit, cam_pitch_limit
            )
        elif key == Keyboard.LEFT:
            cam_yaw_offset = clamp(
                cam_yaw_offset - cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )
        elif key == Keyboard.RIGHT:
            cam_yaw_offset = clamp(
                cam_yaw_offset + cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )

        key = keyboard.getKey()

    # Control inputs
    roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + gx + roll_disturbance
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + gy + pitch_disturbance
    yaw_input = yaw_disturbance

    clamped_diff_alt = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * pow(clamped_diff_alt, 3.0)

    # Motor mixing (note the sign inversions to match propeller directions)
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
