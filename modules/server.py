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
from triangulation import TriangulationBAModule, TriangulationF2FModule
import argparse
import os
import logging
from datetime import datetime
import time
import queue
from threading import Thread
from typing import List
from collections import deque

logging.basicConfig(
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

    is_motion = (m >= threshold_mean) or (h >= threshold_hist) or (s <= threshold_ssim)
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


def recv_metadata(sock: socket.socket) -> List[np.ndarray]:
    header = recv_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    if length == 0 or length > 50_000_000:
        raise ValueError(f"Bad frame length: {length}")
    payload = recv_exact(sock, length)
    metadata = json.loads(payload.decode("utf-8"))
    if metadata is None:
        raise ValueError("Failed to decode metadata")
    K = np.array(metadata["K"]).reshape(3, 3)
    R = np.array(metadata["R"]).reshape(3, 3)
    T = np.array(metadata["T"]).reshape(3, 1)
    metadata = {"K": K, "R": R, "T": T}
    return metadata


def process_stream_async(output_path: str, conn: socket.socket, addr: str):
    try:
        logging.info(f"[+] Connected: {addr}")
        now = datetime.now()
        time_str = now.strftime("%H_%M_%S")
        output_path = os.path.join(output_path, time_str)
        os.makedirs(output_path, exist_ok=True)

        yolo = YoloDetectionMain(conf_threshold=CONF)
        logging.info("YoloV8 model instantiated")
        sort_manager = SORTTrackManager()
        logging.info("SORT manager instantiated")

        json_path = os.path.join(output_path, f"stream_{time_str}.json")
        cpu_csv_path = os.path.join(output_path, f"stream_{time_str}.csv")
        cvdata_json_path = os.path.join(output_path, f"stream_{time_str}_cv.json")
        label_mp4_path = os.path.join(output_path, f"stream_{time_str}_labeled.mp4")
        raw_mp4_path = os.path.join(output_path, f"stream_{time_str}_raw.mp4")

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
                    "yolo_pct_machine",
                    "yolo_cores",
                ]
            )
        with open(cvdata_json_path, "w") as f:
            f.write("[\n")

        start_timestamp = time.time()

        q = queue.Queue(maxsize=1e6)
        stop_sentinel = object()

        def recv_loop():
            frame_idx = 0
            try:
                while True:
                    frame = recv_frame(conn)
                    metadata = recv_metadata(conn)
                    ts = time.time()
                    item = (frame_idx, frame, metadata, ts)
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
                    q.put_nowait((None, stop_sentinel, None, None))
                except Exception:
                    pass

        def worker_loop():
            frame_buffer = deque(maxlen=100)
            outputs = []
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            label_mp4 = cv2.VideoWriter(
                label_mp4_path,
                fourcc,
                FPS,
                (640, 480),
            )
            raw_mp4 = cv2.VideoWriter(
                raw_mp4_path,
                fourcc,
                FPS,
                (640, 480),
            )
            logging.info("Video Writer instantiated")

            processing_times = {
                "motion_gate": {"total": 0, "count": 0},
                "yolo_detection": {"total": 0, "count": 0},
                "midas_estimation": {"total": 0, "count": 0},
                "sort_tracking": {"total": 0, "count": 0},
                "drawing_on_frame": {"total": 0, "count": 0},
                "triangulation": {"total": 0, "count": 0},
            }

            def update_time(name, start):
                duration = time.time() - start
                processing_times[name]["total"] += duration
                processing_times[name]["count"] += 1

            while True:
                logging.info(f"Total Frames remaining to process: {q.qsize()}")
                idx, frame, metadata, ts = q.get()

                if frame is stop_sentinel:
                    logging.info("End of queue reached. Exiting worker thread.")
                    break
                else:
                    raw_mp4.write(frame)

                force_detection = idx % 10 == 0

                start_time = time.time()
                motion = (
                    True
                    if not frame_buffer
                    else (force_detection or motion_gate(frame, frame_buffer[-1][0]))
                )
                update_time("motion_gate", start_time)

                if not motion:
                    start_time = time.time()
                    outputs = sort_manager.step([], idx)
                    update_time("sort_tracking", start_time)
                else:
                    start_time = time.time()
                    y_results, y_dt, y_pct_all, y_pct_machine, y_cores = (
                        measure_cpu_usage(process, yolo, frame)
                    )
                    update_time("yolo_detection", start_time)

                    timestamp_cpu = ts - start_timestamp
                    with open(cpu_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                timestamp_cpu,
                                y_pct_all,
                                y_pct_machine,
                                y_cores,
                            ]
                        )

                    detection_results = []
                    for obj in y_results:
                        payload = {
                            "frame_idx": int(idx),
                            "class_name": str(obj[0]),
                            "conf": float(obj[1]),
                            "center": np.asarray(obj[2]).tolist(),
                            "bbox": np.asarray(obj[3]).tolist(),
                            "ts_detect": y_dt,
                        }
                        detection_results.append(payload)

                    start_time = time.time()
                    outputs = sort_manager.step(detection_results, idx)
                    update_time("sort_tracking", start_time)

                frame_buffer.append(
                    (frame, metadata["K"], metadata["R"], metadata["T"])
                )

                with open(cvdata_json_path, "a") as f:
                    f.write(",\n")
                    json.dump(
                        {
                            "frame_idx": idx,
                            "data": {
                                "K": metadata["K"].tolist(),
                                "R": metadata["R"].tolist(),
                                "t": metadata["T"].tolist(),
                                "bboxes": [data["bbox"] for data in outputs],
                            },
                        },
                        f,
                        indent=4,
                    )

                start_time = time.time()
                for i, output in enumerate(outputs):
                    with open(json_path, "a") as f:
                        f.write(",\n")
                        json.dump(output, f, indent=4)

                    x1, y1, x2, y2 = map(int, output["bbox"])
                    track_id = output["track_id"]
                    class_name = output["class_name"]
                    if class_name not in CLASS_FILTER:
                        continue

                    color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_name} ID:{track_id} Predict: {output['kalman']}"
                    coords_label = ""

                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )

                    (coords_w, coords_h), baseline2 = cv2.getTextSize(
                        coords_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    total_h = label_h + coords_h + baseline + baseline2
                    total_w = max(label_w, coords_w)

                    cv2.rectangle(
                        frame,
                        (x1, y1 - total_h),
                        (x1 + total_w, y1),
                        color,
                        -1,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - coords_h - baseline2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    if coords_label:
                        cv2.putText(
                            frame,
                            coords_label,
                            (x1, y1 - baseline2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                update_time("drawing_on_frame", start_time)

                label_mp4.write(frame)

            label_mp4.release()
            raw_mp4.release()
            logging.info("Video writer released in worker_loop")
            print("Average processing times:")
            for name, data in processing_times.items():
                if data["count"] > 0:
                    avg_time = data["total"] / data["count"]
                    print(f"{name}: {avg_time:.4f}s")

        def postprocess_loop(cvdata_json_path, raw_mp4_path):
            tri_ba_module = TriangulationBAModule(maxlen=TRI_MIN_SIZE)
            logging.info("Triangulation Bundle Adjustment Module initialized.")

            with open(cvdata_json_path, "r") as f:
                cv_data = json.load(f)

            cap = cv2.VideoCapture(raw_mp4_path)
            if not cap.isOpened():
                logging.error("Could not open video file.")
                return

            frame_idx = 0
            initialized = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_data = next(
                    (item for item in cv_data if item.get("frame_idx") == frame_idx),
                    None,
                )
                if not frame_data:
                    logging.warning(f"No CV data found for frame {frame_idx}")
                    frame_idx += 1
                    continue

                tri_ba_module.insert(
                    np.array(frame_data["K"]),
                    frame,
                    np.array(frame_data["R"]),
                    np.array(frame_data["t"]),
                    frame_data["bboxes"],
                )

                frame_idx += 1

        recv_t = Thread(target=recv_loop, daemon=True)
        work_t = Thread(target=worker_loop, daemon=True)
        recv_t.start()
        work_t.start()

    except Exception as e:
        logging.error(f"Stream crashed: {e}")

    finally:
        logging.info(f"[-] Disconnected: {addr}")
        work_t.join()
        with open(json_path, "a") as f:
            f.write("\n]")
        with open(cvdata_json_path, "a") as f:
            f.write("\n]")
        conn.close()
        postprocess_loop(cvdata_json_path, raw_mp4_path)


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
        "--o", type=str, default="server_output", help="Output directory for json files"
    )

    args = parser.parse_args()

    main(args)
