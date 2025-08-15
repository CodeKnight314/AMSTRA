import cv2
from controller import Robot, Keyboard
from math import pow
import numpy as np
import struct, socket, time


class FrameClient:
    def __init__(self, host: str, port: int, timeout: float, jpeg_quality: int):
        self.address = (host, port)
        self.timeout = timeout
        assert (
            jpeg_quality >= 0 and jpeg_quality <= 100
        ), "JPEG Quality must be between 0 and 100."
        self.quality = jpeg_quality
        self.sock = None
        self._connect()

    def send_frame(self, bgr: np.ndarray):
        ok, enc = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if not ok:
            return
        data = enc.tobytes()
        if len(data) == 0 or len(data) > 50_000_000:
            return
        header = struct.pack("!I", len(data))
        try:
            self.sock.sendall(header)
            self.sock.sendall(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            try:
                print(
                    "Restarting connection due to BrokenPipeError, ConnecetionResetError, OSError or other Exception"
                )
                time.sleep(0.25)
                self._connect()
                self.sock.sendall(header)
                self.sock.sendall(data)
            except Exception:
                pass

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None

    def _connect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.address)
        s.settimeout(None)
        self.sock = s
        print(f"[client] connected to {self.address[0]}:{self.address[1]}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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

print("Controls:")
print("- WASD: move (W/S=fwd/back, A/D=strafe)")
print("- Q/E: yaw left/right")
print("- R/F: altitude up/down")
print("- Arrow keys: control camera (Left/Right=yaw, Up/Down=pitch)")

k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

target_altitude = 1.0

cam_pitch_offset = 0.0
cam_yaw_offset = 0.0
cam_pitch_step = 0.02
cam_yaw_step = 0.03
cam_pitch_limit = 0.5
cam_yaw_limit = 1.6

while robot.step(timestep) != -1:
    now = robot.getTime()

    roll, pitch, _yaw = imu.getRollPitchYaw()
    gx, gy, _gz = gyro.getValues()
    _x, _y, altitude = gps.getValues()

    frame_bgr = camera_frame_bgr()
    if frame_bgr is not None:
        sender.send_frame(frame_bgr)

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
                cam_yaw_offset + cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )
        elif key == Keyboard.RIGHT:
            cam_yaw_offset = clamp(
                cam_yaw_offset - cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )

        key = keyboard.getKey()

    pitch_input = (
        k_pitch_p * clamp(pitch - initial_pitch, -1.0, 1.0) + gy + pitch_disturbance
    )
    roll_input = (
        k_roll_p * clamp(roll - initial_roll, -1.0, 1.0) + gx + roll_disturbance
    )
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
