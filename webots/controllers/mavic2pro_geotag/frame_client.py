from queue import Queue, Empty, Full
import socket
import threading, struct
import numpy as np
from utils import transmission_delay
import cv2
import time
import json


class FrameClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        jpeg_quality: int,
        max_queue: int = 100,
        drop_policy: str = "drop_oldest",
    ):
        assert 0 <= jpeg_quality <= 100
        assert drop_policy in ("drop_oldest", "drop_newest")
        self.address = (host, port)
        self.timeout = timeout
        self.quality = jpeg_quality
        self.q = Queue(maxsize=max_queue)
        self.drop_policy = drop_policy
        self._stop = threading.Event()
        self.sock = None
        self._connect()

        self._tx = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx.start()

    def send_frame(self, bgr: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
        while True:
            try:
                batch = (bgr, K, R, t)
                self.q.put_nowait(batch)
                break
            except Full:
                if self.drop_policy == "drop_oldest":
                    try:
                        self.q.get_nowait()
                    except Empty:
                        pass
                else:
                    return

    def close(self):
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except Full:
            pass
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

    def _tx_loop(self):
        while not self._stop.is_set():
            transmission_delay()
            item = self.q.get()
            if item is None:
                break
            bgr, k, r, t = item
            ok, enc = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if not ok:
                continue
            data = enc.tobytes()
            if len(data) == 0 or len(data) > 50_000_000:
                continue
            header = struct.pack("!I", len(data))
            try:
                self.sock.sendall(header)
                self.sock.sendall(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                try:
                    print("TX error → reconnecting…")
                    time.sleep(0.25)
                    self._connect()
                    self.sock.sendall(header)
                    self.sock.sendall(data)
                except Exception:
                    pass

            metadata = {
                "K": k.flatten().tolist(),
                "R": r.flatten().tolist(),
                "T": t.flatten().tolist(),
            }

            meta_bytes = json.dumps(metadata).encode("utf-8")
            meta_header = struct.pack("!I", len(meta_bytes))
            try:
                self.sock.sendall(meta_header)
                self.sock.sendall(meta_bytes)
            except (BrokenPipeError, ConnectionResetError, OSError):
                try:
                    print("TX error → reconnecting...")
                    time.sleep(1)
                    self._connect()
                    self.sock.sendall(meta_header)
                    self.sock.sendall(meta_bytes)
                except Exception:
                    pass
