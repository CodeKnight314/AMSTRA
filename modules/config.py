import psutil

THRESHOLD_MEAN = 0.02
THRESHOLD_HIST = 0.15
THRESHOLD_SSIM = 0.90

CONF = 0.75
N_BASE = 10
FPS = 60

HOST = "127.0.0.1"
PORT = 5001
BACKLOG = 4

N_LOGICAL = psutil.cpu_count(logical=True) or 1

CLASS_FILTER = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "animal"]

TRI_MIN_SIZE = 100

MIN_RECV_SIZE = 0
MAX_RECV_SIZE = 50_000_000

# For short-term memory to use @ motion-gate
FRAME_SHORT_BUFFER_SIZE = 1e2
# For long-term memory to prevent server gets overwhelmed
FRAME_LONG_BUFFER_SIZE = 1e6

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

MIN_SWEEP_INTERVAL = 10

DETECTION_BOX_COLOR = (0, 255, 0)
DETECTION_BOX_THICKNESS = 2
