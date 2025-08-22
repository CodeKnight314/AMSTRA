import psutil

THRESHOLD_MEAN = 0.02
THRESHOLD_HIST = 0.15
THRESHOLD_SSIM = 0.90

CONF = 0.75
N_BASE = 10
FPS = 60
FULL_SWEEP_PERIOD = 4 * FPS

HOST = "127.0.0.1"
PORT = 5001
BACKLOG = 4

N_LOGICAL = psutil.cpu_count(logical=True) or 1

CLASS_FILTER = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "animal"]

TRI_MIN_SIZE = 50
