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
from sort_tracking import SORTTrackManager
from dashboard import Dashboard
import argparse
import os
from datetime import datetime
import time
import queue
from threading import Thread
from typing import List
from collections import deque

dashboard = Dashboard(host=HOST, port=PORT)


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
        stream_id = f"{addr[0]}:{addr[1]}"
        dashboard.update_stream(
            stream_id,
            frames=0,
            queue=0,
            cpu=0,
            status="Connected",
            time_created=datetime.now(),
        )
        now = datetime.now()
        time_str = now.strftime("%H_%M_%S")
        output_path = os.path.join(output_path, time_str)
        os.makedirs(output_path, exist_ok=True)

        yolo = YoloDetectionMain(conf_threshold=CONF)
        sort_manager = SORTTrackManager()

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
                pass
            finally:
                try:
                    q.put_nowait((None, stop_sentinel, None, None))
                except Exception:
                    pass

        def worker_loop():
            firstNavEntry = True
            firstOutputEntry = True
            frame_buffer = deque(maxlen=1000)
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

            while True:
                idx, frame, metadata, ts = q.get()

                if frame is stop_sentinel:
                    break
                else:
                    raw_mp4.write(frame)

                force_detection = idx % 10 == 0

                motion = (
                    True
                    if not frame_buffer
                    else (force_detection or motion_gate(frame, frame_buffer[-1][0]))
                )

                if not motion:
                    outputs = sort_manager.step([], idx)
                else:
                    y_results, y_dt, y_pct_all, y_pct_machine, y_cores = (
                        measure_cpu_usage(process, yolo, frame)
                    )

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

                    outputs = sort_manager.step(detection_results, idx)

                frame_buffer.append(
                    (frame, metadata["K"], metadata["R"], metadata["T"])
                )

                with open(cvdata_json_path, "a") as f:
                    if not firstNavEntry:
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
                    firstNavEntry = False

                dashboard.update_stream(
                    stream_id,
                    frames=idx,
                    queue=q.qsize(),
                    cpu=round(process.cpu_percent(interval=None) / N_LOGICAL, 2),
                    status="Running",
                )

                for i, output in enumerate(outputs):
                    with open(json_path, "a") as f:
                        if not firstOutputEntry:
                            f.write(",\n")
                        json.dump(output, f, indent=4)
                        firstOutputEntry = False

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

                label_mp4.write(frame)

            label_mp4.release()
            raw_mp4.release()

        recv_t = Thread(target=recv_loop, daemon=True)
        work_t = Thread(target=worker_loop, daemon=True)
        recv_t.start()
        work_t.start()

    except Exception as e:
        dashboard.update_stream(stream_id, status="Crashed", cpu=0, queue=0)

    finally:
        work_t.join()
        with open(json_path, "a") as f:
            f.write("\n]")
        with open(cvdata_json_path, "a") as f:
            f.write("\n]")
        dashboard.update_stream(stream_id, status="Disconnected", cpu=0, queue=0)
        clear_terminal()
        conn.close()


def start_server(output_path: str, host=HOST, port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(BACKLOG)
        while True:
            conn, addr = s.accept()
            t = threading.Thread(
                target=process_stream_async, args=(output_path, conn, addr), daemon=True
            )
            t.start()


def main(args):
    dash_thread = threading.Thread(target=dashboard.run, daemon=True)
    dash_thread.start()
    start_server(output_path=args.o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated Ground Station Server")
    parser.add_argument(
        "--o", type=str, default="server_output", help="Output directory for json files"
    )

    args = parser.parse_args()

    main(args)
