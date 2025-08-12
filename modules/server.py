import socket
import struct
import threading
import json
from utils import *
from config import *
import numpy as np
import cv2
from typing import Tuple
from skimage.metrics import structural_similarity as ssim
from yolo_detector import YoloDetection
from midas_depth_estimation import MiDasEstimation
import argparse
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

HOST = "127.0.0.1"
PORT = 5001
BACKLOG = 4


def img_postprocess(image: np.array, dim: Tuple[int, int] = (64, 36)):
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return grayscale


def mean_abs_diff(curr: np.array, ref: np.array):
    if curr.shape != ref.shape:
        raise ValueError(
            f"Current array does not match reference array in shape. Expected {ref.shape} from current array but got {curr.shape}"
        )
    c = curr.astype(np.float32) / 255.0
    r = ref.astype(np.float32) / 255.0
    return float(np.mean(np.abs(c - r)))


def hist_diff(curr: np.array, ref: np.array):
    cur_hist = cv2.calcHist([curr], [0], None, [64], [0, 255])
    ref_hist = cv2.calcHist([ref], [0], None, [64], [0, 255])
    cur_hist = cv2.normalize(cur_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    ref_hist = cv2.normalize(ref_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(cur_hist, ref_hist, cv2.HISTCMP_CHISQR))


def ssim_diff(curr: np.ndarray, ref: np.ndarray) -> float:
    c = curr.astype(np.float32) / 255.0
    r = ref.astype(np.float32) / 255.0
    dr = float(max(1e-6, c.max() - c.min(), r.max() - r.min()))
    score, _ = ssim(c, r, full=True, data_range=dr)
    return float(score)


def motion_gate(
    curr: np.ndarray,
    prev: np.ndarray,
    threshold_mean: float = THRESHOLD_MEAN,
    threshold_hist: float = THRESHOLD_HIST,
    threshold_ssim: float = THRESHOLD_SSIM,
    early_exit: bool = True,
) -> bool:

    curr_gs = img_postprocess(curr)
    prev_gs = img_postprocess(prev)

    m = mean_abs_diff(curr_gs, prev_gs)

    if early_exit and m < 0.005:
        s = ssim_diff(curr_gs, prev_gs)
        return s <= 0.98

    h = hist_diff(curr_gs, prev_gs)
    s = ssim_diff(curr_gs, prev_gs)

    is_motion = (
        (m >= threshold_mean) and (h >= threshold_hist) and (s <= threshold_ssim)
    )
    return is_motion


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buff = bytearray()
    while len(buff) < n:
        chunk = sock.recv(n - len(buff))
        if not chunk:
            raise ConnectionError("Socket closed during recv_exact")
        buff.extend(chunk)
    return bytes(buff)


def recv_frame(sock: socket.socket) -> np.ndarray:
    header = recv_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    if length == 0 or length > 50_000_000:
        raise ValueError(f"Bad frame length: {length}")
    payload = recv_exact(sock, length)
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG")
    return img


def process_stream(output_path: str, conn: socket.socket, addr: str):
    logging.info(f"[+] Connected: {addr}")
    os.makedirs(output_path, exist_ok=True)
    yolo = YoloDetection(conf_threshold=CONF)
    logging.info("YoloV8 model instantiated")
    midas = MiDasEstimation()
    logging.info("MiDaS model instantiated")

    prev_frame = None
    frame_count = 0

    now = datetime.now()
    time_str = now.strftime("%H_%M_%S")
    json_path = os.path.join(output_path, f"stream_{time_str}.json")
    all_results = []

    try:
        while True:
            logging.info(f"Attempting to retrieve frame")
            curr_frame = recv_frame(conn)
            logging.info(f"Successfully retrieved frame")

            if prev_frame is None:
                prev_frame = curr_frame
                motion = True
            else:
                motion = motion_gate(curr_frame, prev_frame)
            logging.info(f"Motion Detected: {motion}")
            if motion:
                detection_results = yolo(curr_frame, return_boxes=True)
                depth_map = midas(curr_frame)

                for obj in detection_results:
                    x, y = map(int, obj[2])
                    h, w = depth_map.shape[:2]
                    x = np.clip(x, 0, w - 1)
                    y = np.clip(y, 0, h - 1)
                    z = float(depth_map[y, x])

                    payload = {
                        "frame_idx": int(frame_count),
                        "class_name": str(obj[0]),
                        "conf": float(obj[1]),
                        "center": np.asarray(obj[2]).tolist(),
                        "bbox": np.asarray(obj[3]).tolist(),
                        "depth_value": float(z),
                    }

                    all_results.append(payload)

                with open(json_path, "w") as f:
                    json.dump(all_results, f, indent=4)

            frame_count += 1
            prev_frame = curr_frame

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        conn.close()
        logging.info(f"[-] Disconnected: {addr}")


def start_server(output_path: str, host=HOST, port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(BACKLOG)
        logging.info(f"🚦 Listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            t = threading.Thread(
                target=process_stream, args=(output_path, conn, addr), daemon=True
            )
            t.start()


def main(args):
    start_server(output_path=args.o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated Ground Station Server")
    parser.add_argument(
        "--o", type=str, required=True, help="Output directory for json files"
    )

    args = parser.parse_args()

    main(args)
