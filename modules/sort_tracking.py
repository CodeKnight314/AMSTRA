from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
import math


class KalmanFilter:
    def __init__(self, initial_state: np.ndarray):
        self.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.x = initial_state

        self.P = np.eye(7) * 500.0

        self.Q = np.eye(7)

        self.R = np.eye(4) * 10.0

        self.I = np.eye(7)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement: np.ndarray):
        y = measurement - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (self.I - (K @ self.H)) @ self.P
        return self.x

    def get_state(self) -> np.ndarray:
        return self.x


def bbox_to_state(bbox: np.ndarray):
    measurement = bbox_to_measurement(bbox)
    return np.vstack([measurement, np.zeros((3, 1))], dtype=np.float32)


def bbox_to_measurement(bbox: np.ndarray):
    x1, y1, x2, y2 = map(float, bbox)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([[u], [v], [s], [r]], dtype=float)


def state_to_bbox(state: np.ndarray):
    u, v, s, r = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    w = math.sqrt(r * s)
    h = s / w
    x1 = u - w / 2
    x2 = u + w / 2
    y1 = v - h / 2
    y2 = v + h / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou(bbox_1: np.ndarray, bbox_2: np.ndarray):
    x11, y11, x12, y12 = (
        float(bbox_1[0]),
        float(bbox_1[1]),
        float(bbox_1[2]),
        float(bbox_1[3]),
    )

    x21, y21, x22, y22 = (
        float(bbox_2[0]),
        float(bbox_2[1]),
        float(bbox_2[2]),
        float(bbox_2[3]),
    )

    inter_w = max(0, min(x12, x22) - max(x11, x21))
    inter_h = max(0, min(y12, y22) - max(y11, y21))
    inter_area = inter_h * inter_w

    total_area = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - inter_area

    iou = inter_area / total_area
    return iou


class Track:
    def __init__(self, initial_payload: dict, track_id: int):
        self.track_id = track_id
        self.filter = KalmanFilter(initial_state=bbox_to_state(initial_payload["bbox"]))

        self.payload = dict(initial_payload)
        self.payload["track_id"] = self.track_id
        self.payload["kalman"] = False

        self.time_since_update = 0
        self.hit_streak = 1
        self.age = 1

    def predict(self, frame_idx: int):
        self.age += 1
        self.time_since_update += 1
        state = self.filter.predict()
        bbox_pred = state_to_bbox(state).tolist()
        u, v = float(state[0]), float(state[1])

        payload = dict(self.payload)
        payload.update(
            {
                "frame_idx": frame_idx,
                "center": (u, v),
                "bbox": bbox_pred,
                "conf": None,
                "depth_value": None,
                "ts_detect": None,
                "ts_depth": None,
            }
        )
        payload["kalman"] = True
        return payload

    def update(self, new_payload: dict):
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        z = bbox_to_measurement(new_payload["bbox"])
        self.filter.update(z)

        self.payload = dict(new_payload)
        self.payload["track_id"] = self.track_id
        self.payload["kalman"] = False
        return dict(self.payload)

    def get_track_id(self):
        return self.track_id

    def get_bbox(self):
        return state_to_bbox(self.filter.get_state())

    def get_state(self):
        return self.filter.get_state()


class SORTTrackManager:
    def __init__(self, max_age: int = 10, min_age: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_age = min_age
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.curr_track_id = 0

    def step(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        output = []

        matches = []
        unmatched_tracks = []
        unmatched_dets = []

        if not self.tracks:
            for det in detections:
                new_track = Track(det, self.curr_track_id)
                self.curr_track_id += 1
                self.tracks.append(new_track)
                output.append(dict(new_track.payload))
            return output

        if not detections:
            for track in self.tracks:
                prediction = track.predict(frame_idx)
                if track.hit_streak >= self.min_age:
                    output.append(prediction)
            self.tracks = [
                t for t in self.tracks if t.time_since_update <= self.max_age
            ]
            return output

        if self.tracks and detections is not None and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

            for t_idx, track in enumerate(self.tracks):
                pred_bbox = track.get_bbox()
                for d_idx, detection in enumerate(detections):
                    detected_bbox = detection["bbox"]
                    iou_matrix[t_idx, d_idx] = iou(pred_bbox, detected_bbox)

            rows, cols = linear_sum_assignment(-iou_matrix)
            assigned_tracks = set()
            assigned_detection = set()
            for r, c in zip(rows, cols):
                if iou_matrix[r, c] > self.iou_threshold:
                    matches.append((r, c))
                    assigned_tracks.add(r)
                    assigned_detection.add(c)

            for t_idx, track in enumerate(self.tracks):
                if t_idx not in assigned_tracks:
                    unmatched_tracks.append(t_idx)

            for d_idx, detection in enumerate(detections):
                if d_idx not in assigned_detection:
                    unmatched_dets.append(d_idx)

        for t_idx, d_idx in matches:
            payload = self.tracks[t_idx].update(detections[d_idx])
            output.append(payload)

        for idx in unmatched_tracks:
            track = self.tracks[idx]
            prediction = track.predict(frame_idx)
            if track.hit_streak >= self.min_age:
                output.append(prediction)

        for idx in unmatched_dets:
            new_track = Track(detections[idx], self.curr_track_id)
            self.curr_track_id += 1
            self.tracks.append(new_track)
            output.append(dict(new_track.payload))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return output
