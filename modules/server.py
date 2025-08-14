import socket
import psutil
import csv
import struct
import threading
import json
from utils import *
from config import *
import numpy as np
import cv2
from yolo_detector import YoloDetection
from midas_depth_estimation import MiDasEstimation
from sort_tracking import SORTTrackManager
import argparse
import os
import logging
from datetime import datetime
import time

logging.basicConfig(
    filename=f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def motion_gate(
    curr: np.ndarray,
    prev: np.ndarray,
    threshold_mean: float = THRESHOLD_MEAN,
    threshold_hist: float = THRESHOLD_HIST,
    threshold_ssim: float = THRESHOLD_SSIM,
    early_exit: bool = True,
) -> bool:

    curr_gs = img_preprocess(curr)
    prev_gs = img_preprocess(prev)

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
    sort_manager = SORTTrackManager()
    logging.info("SORT manager instantiated")

    prev_frame = None
    frame_count = 0

    now = datetime.now()
    time_str = now.strftime("%H_%M_%S")
    json_path = os.path.join(output_path, f"stream_{time_str}.json")
    cpu_csv_path = os.path.join(output_path, f"stream_{time_str}.csv")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    with open(json_path, "w") as f:
        f.write("[\n")

    with open(cpu_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "yolo_pct_all",
                "depth_pct_all",
                "yolo_pct_machine",
                "depth_pct_machine",
                "yolo_cores",
                "depth_cores",
            ]
        )

    first_entry = True

    start_timestamp = time.time()

    try:
        while True:
            curr_frame = recv_frame(conn)

            if prev_frame is None:
                prev_frame = curr_frame
                motion = True
            else:
                motion = motion_gate(curr_frame, prev_frame)

            if motion:
                logging.info(f"Motion detected from camera feed")
                y_results, y_dt, y_pct_all, y_pct_machine, y_cores = measure_cpu_usage(
                    process, yolo, curr_frame
                )

                d_results, d_dt, d_pct_all, d_pct_machine, d_cores = measure_cpu_usage(
                    process, midas, curr_frame
                )

                timestamp_cpu = time.time() - start_timestamp
                with open(cpu_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            timestamp_cpu,
                            y_pct_all,
                            d_pct_all,
                            y_pct_machine,
                            d_pct_machine,
                            y_cores,
                            d_cores,
                        ]
                    )

                detection_results = []
                for obj in y_results:
                    x, y = map(int, obj[2])
                    h, w = d_results.shape[:2]
                    x = np.clip(x, 0, w - 1)
                    y = np.clip(y, 0, h - 1)
                    z = float(d_results[y, x])

                    payload = {
                        "frame_idx": int(frame_count),
                        "class_name": str(obj[0]),
                        "conf": float(obj[1]),
                        "center": np.asarray(obj[2]).tolist(),
                        "bbox": np.asarray(obj[3]).tolist(),
                        "rel_depth_value": float(z),
                        "ts_detect": y_dt,
                        "ts_depth": d_dt,
                    }

                    detection_results.append(payload)

                outputs = sort_manager.step(detection_results, frame_count)
                print(len(outputs))

                for output in outputs:
                    with open(json_path, "a") as f:
                        if not first_entry:
                            f.write(",\n")
                        json.dump(output, f, indent=4)
                        first_entry = False

            frame_count += 1
            prev_frame = curr_frame

    except Exception as e:
        import traceback

        traceback.print_exc()
        logging.error(f"An error occurred: {e}")

    finally:
        with open(json_path, "a") as f:
            f.write("\n]")
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
