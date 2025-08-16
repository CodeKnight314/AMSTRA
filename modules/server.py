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
from yolo_detector import YoloDetectionMain
from midas_depth_estimation import MiDasEstimation
from sort_tracking import SORTTrackManager
import argparse
import os
import logging
from datetime import datetime
import time
import queue
from threading import Thread

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

    if early_exit and m < 0.001:
        s = ssim_diff(curr_gs, prev_gs)
        return s <= 0.99

    h = hist_diff(curr_gs, prev_gs)
    s = ssim_diff(curr_gs, prev_gs)

    is_motion = (
        (m >= threshold_mean) or (h >= threshold_hist) or (s <= threshold_ssim)
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


def process_stream_sync(output_path: str, conn: socket.socket, addr: str):
    logging.info(f"[+] Connected: {addr}")
    now = datetime.now()
    time_str = now.strftime("%H_%M_%S")
    output_path = os.path.join(output_path, time_str)
    os.makedirs(output_path, exist_ok=True)
    yolo = YoloDetectionMain(conf_threshold=CONF)
    logging.info("YoloV8 model instantiated")
    midas = MiDasEstimation()
    logging.info("MiDaS model instantiated")
    sort_manager = SORTTrackManager()
    logging.info("SORT manager instantiated")

    prev_frame = None
    frame_count = 0
    json_path = os.path.join(output_path, f"stream_{time_str}.json")
    cpu_csv_path = os.path.join(output_path, f"stream_{time_str}.csv")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(output_path, f"stream_{time_str}.mp4"), fourcc, FPS, (640, 480)
    )
    logging.info("Video Writer instantiated")

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
        outputs = []
        while True:
            curr_frame = recv_frame(conn)

            if prev_frame is None:
                prev_frame = curr_frame
                motion = True
            else:
                motion = motion_gate(curr_frame, prev_frame)

            if not motion:
                outputs = sort_manager.step([], frame_count)
            else:
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
                print("Detection Result Length: ", len(detection_results))
                outputs = sort_manager.step(detection_results, frame_count)
                print("SORT Manager Output Length: ", len(outputs))

            for output in outputs:
                with open(json_path, "a") as f:
                    if not first_entry:
                        f.write(",\n")
                    json.dump(output, f, indent=4)
                    first_entry = False

                x1, y1, x2, y2 = map(int, output["bbox"])
                track_id = output["track_id"]
                class_name = output["class_name"]

                cv2.rectangle(curr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} ID:{track_id}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    curr_frame,
                    (x1, y1 - label_h - baseline),
                    (x1 + label_w, y1),
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    curr_frame,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            out.write(curr_frame)

            frame_count += 1
            prev_frame = curr_frame

    except Exception as e:
        import traceback

        traceback.print_exc()
        logging.error(f"An error occurred: {e}")

    finally:
        with open(json_path, "a") as f:
            f.write("\n]")
        out.release()

        conn.close()
        logging.info(f"[-] Disconnected: {addr}")


def process_stream_async(output_path: str, conn: socket.socket, addr: str):
    logging.info(f"[+] Connected: {addr}")
    now = datetime.now()
    time_str = now.strftime("%H_%M_%S")
    output_path = os.path.join(output_path, time_str)
    os.makedirs(output_path, exist_ok=True)

    yolo = YoloDetectionMain(conf_threshold=CONF)
    logging.info("YoloV8 model instantiated")
    midas = MiDasEstimation()
    logging.info("MiDaS model instantiated")
    sort_manager = SORTTrackManager()
    logging.info("SORT manager instantiated")

    json_path = os.path.join(output_path, f"stream_{time_str}.json")
    cpu_csv_path = os.path.join(output_path, f"stream_{time_str}.csv")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(output_path, f"stream_{time_str}.mp4"), fourcc, FPS, (640, 480)
    )
    logging.info("Video Writer instantiated")

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

    start_timestamp = time.time()
    first_entry = True

    q = queue.Queue(maxsize=8)
    stop_sentinel = object()

    def recv_loop():
        frame_idx = 0
        try:
            while True: 
                frame = recv_frame(conn)
                ts = time.time()
                item = (frame_idx, frame, ts)
                while True:
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                frame_idx += 1
        except Exception as e:
            logging.info(f"recv_loop ending: {e}")
        finally:
            try:
                q.put_nowait((None, stop_sentinel, None))
            except Exception:
                pass

    def worker_loop():
        nonlocal first_entry
        prev_frame = None
        outputs = []
        while True:
            idx, frame, ts = q.get()
            if frame is stop_sentinel:
                break

            force_detection = (idx % 10 == 0)
            motion = True if prev_frame is None else (force_detection or motion_gate(frame, prev_frame))

            if not motion:
                outputs = sort_manager.step([], idx)
            else:
                y_results, y_dt, y_pct_all, y_pct_machine, y_cores = measure_cpu_usage(
                    process, yolo, frame
                )
                d_results, d_dt, d_pct_all, d_pct_machine, d_cores = measure_cpu_usage(
                    process, midas, frame
                )

                timestamp_cpu = ts - start_timestamp
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
                h, w = d_results.shape[:2]
                for obj in y_results:
                    x, y = map(int, obj[2])
                    x = np.clip(x, 0, w - 1)
                    y = np.clip(y, 0, h - 1)
                    z = float(d_results[y, x])
                    payload = {
                        "frame_idx": int(idx),
                        "class_name": str(obj[0]),
                        "conf": float(obj[1]),
                        "center": np.asarray(obj[2]).tolist(),
                        "bbox": np.asarray(obj[3]).tolist(),
                        "rel_depth_value": z,
                        "ts_detect": y_dt,
                        "ts_depth": d_dt,
                    }
                    detection_results.append(payload)

                outputs = sort_manager.step(detection_results, idx)

            for output in outputs:
                with open(json_path, "a") as f:
                    if not first_entry:
                        f.write(",\n")
                    json.dump(output, f, indent=4)
                    first_entry = False

                x1, y1, x2, y2 = map(int, output["bbox"])
                track_id = output["track_id"]
                class_name = output["class_name"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} ID:{track_id}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_h - baseline),
                    (x1 + label_w, y1),
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            print(len(outputs))
            out.write(frame)

            prev_frame = frame

    recv_t = Thread(target=recv_loop, daemon=True)
    work_t = Thread(target=worker_loop, daemon=True)
    recv_t.start()
    work_t.start()
    work_t.join()

    try:
        with open(json_path, "a") as f:
            f.write("\n]")
    finally:
        out.release()
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
                target=process_stream_async, args=(output_path, conn, addr), daemon=True
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
