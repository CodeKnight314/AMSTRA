import numpy as np
import cv2
from typing import Tuple, List
from collections import deque
from scipy.optimize import least_squares
import json
import logging
import argparse
from tqdm import tqdm
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class TriangulationF2FModule:
    def __init__(self, k: np.ndarray):
        self.k = k
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._buffer = deque(maxlen=100)

    def __call__(self, frame1, R1, t1, frame2, R2, t2, bboxes):
        return self.infer(frame1, R1, t1, frame2, R2, t2, bboxes)

    def _make_mask(self, shape_hw, bboxes):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 > x1 and y2 > y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 255

        return mask

    def insert(
        self,
        K: np.ndarray,
        frame: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        bboxes: List[Tuple[int]],
    ):
        self.k = K
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_mask = self._make_mask(frame, bboxes)
        keypoints, descriptors = self.orb.detectAndCompute(frame, frame_mask)

        if len(keypoints) < 8 or descriptors is None:
            transition = {
                "K": K,
                "R": R,
                "t": t,
                "keypoints": [],
                "descriptors": None,
            }

        else:
            transition = {
                "K": K,
                "R": R,
                "t": t,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }

        self._buffer.append(transition)

    def _feature_match(self, kp1, des1, kp2, des2):
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.k, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if mask is not None:
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

        return pts1, pts2

    def infer(self, bboxes: List[Tuple[int]]):
        if len(self._buffer) < 2:
            raise ValueError(
                f"Length of internal buffer is not long enough for inference"
            )

        candidate1 = self._buffer[0]
        candidate2 = self._buffer[-1]

        R1, t1 = candidate1["R"], candidate1["t"]
        R2, t2 = candidate2["R"], candidate2["t"]

        P1 = self.k @ np.hstack((R1, t1))
        P2 = self.k @ np.hstack((R2, t2))

        pts1, pts2 = self._feature_match(
            candidate1["keypoints"],
            candidate1["descriptors"],
            candidate2["keypoints"],
            candidate2["descriptors"],
        )
        if len(pts1) == 0 or len(pts2) == 0:
            return {i: [] for i in range(len(bboxes))}

        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        X3D = (pts4D[:3] / pts4D[3]).T

        bbox_to_pts = {i: [] for i in range(len(bboxes))}
        for pt1, pt2, X in zip(pts1, pts2, X3D):
            for i, bbox in enumerate(bboxes):
                if self._is_in_bbox(pt1, [bbox]) and self._is_in_bbox(pt2, [bbox]):
                    bbox_to_pts[i].append(X)

        return bbox_to_pts

    def _is_in_bbox(self, pts: Tuple[float, float], bboxes: List[Tuple[int]]):
        x, y = pts
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True

        return False


class TriangulationBAModule:
    def __init__(self, maxlen: int = 1000):
        self.orb = cv2.ORB_create(nfeatures=512, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._buffer = deque(maxlen=maxlen)
        self.tracks = []
        self.points3D = []

    def _is_in_bbox(self, pts: Tuple[float, float], bboxes: List[Tuple[int]]):
        x, y = pts
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True

        return False

    def _make_mask(self, shape, bboxes: List[Tuple[int]]):
        if bboxes is None:
            return None
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 > x1 and y2 > y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 255
        return mask

    def insert(
        self,
        K: np.ndarray,
        frame: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        bboxes: List[Tuple[int]],
    ):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_mask = self._make_mask(frame.shape, bboxes)
        keypoints, descriptors = self.orb.detectAndCompute(frame, frame_mask)

        if len(keypoints) < 8 or descriptors is None:
            transition = {
                "K": K,
                "frame": frame,
                "R": R,
                "t": t,
                "keypoints": [],
                "descriptors": None,
            }

        else:
            transition = {
                "K": K,
                "frame": frame,
                "R": R,
                "t": t,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }

        self._buffer.append(transition)

        while len(self.points3D) < len(self.tracks):
            self.points3D.append(None)

        if len(self._buffer) >= 2:
            self.update_tracks(len(self._buffer) - 2, len(self._buffer) - 1)

    def _skew_symmetric(self, v):
        v = v.reshape(
            3,
        )
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _findEssentialMat(
        self, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray
    ):
        R_relative = R2 @ R1.T
        t_relative = t2 - R_relative @ t1
        E = self._skew_symmetric(t_relative) @ R_relative
        return E

    def _sampson_inliers(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
        thresh: float = 1.0,
    ):
        Kinv = np.linalg.inv(K)
        x1 = (Kinv @ np.c_[pts1, np.ones(len(pts1))].T).T
        x2 = (Kinv @ np.c_[pts2, np.ones(len(pts2))].T).T
        Ex1 = (E @ x1.T).T
        ETx2 = (E.T @ x2.T).T
        x2Ex1 = np.sum(x2 * (E @ x1.T).T, axis=1)
        denom = (
            Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + ETx2[:, 0] ** 2 + ETx2[:, 1] ** 2 + 1e-12
        )
        d = (x2Ex1**2) / denom
        return d < (thresh**2)

    def match_frames(self, idx1: int, idx2: int):
        f1, f2 = self._buffer[idx1], self._buffer[idx2]
        if f1["descriptors"] is None or f2["descriptors"] is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), []

        matches = self.bf.match(f1["descriptors"], f2["descriptors"])
        if matches is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), []
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([f1["keypoints"][m.queryIdx].pt for m in matches])
        pts2 = np.float32([f2["keypoints"][m.trainIdx].pt for m in matches])

        E = self._findEssentialMat(f1["R"], f1["t"], f2["R"], f2["t"])
        mask = self._sampson_inliers(pts1, pts2, f1["K"], E)

        if mask is not None:
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

        return pts1, pts2, matches

    def _triangulate_two_views(self, pts1, pts2, f1, f2):
        P1 = f1["K"] @ np.hstack((f1["R"], f1["t"]))
        P2 = f2["K"] @ np.hstack((f2["R"], f2["t"]))

        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        X3D = (pts4D[:3] / pts4D[3]).T
        return X3D

    def update_tracks(self, idx1: int, idx2: int):
        pts1, pts2, matches = self.match_frames(idx1, idx2)

        for (u1, v1), (u2, v2) in zip(pts1, pts2):
            matched_track = None
            matched_track_idx = None
            for i, track in enumerate(self.tracks):
                if (
                    idx1 in track
                    and np.linalg.norm(np.array(track[idx1]) - np.array([u1, v1])) < 2.0
                ):
                    matched_track = track
                    matched_track_idx = i
                    break

            if matched_track is not None:
                self.tracks[matched_track_idx][idx2] = (u2, v2)
            else:
                new_track = {idx1: (u1, v1), idx2: (u2, v2)}

                f1, f2 = self._buffer[idx1], self._buffer[idx2]

                pts1 = np.array([[u1, v1]], dtype=np.float32)
                pts2 = np.array([[u2, v2]], dtype=np.float32)
                X = self._triangulate_two_views(pts1, pts2, f1, f2)
                mask = self._filter(f1, f2, X)

                pts1 = pts1[mask]
                pts2 = pts2[mask]
                X = X[mask]

                if mask[0]:
                    self.tracks.append(new_track)
                    self.points3D.append(X[0])

    def initialize_3D_points(self):
        while len(self.points3D) < len(self.tracks):
            self.points3D.append(None)

        for i, track in enumerate(self.tracks):
            if len(track) < 2:
                continue

            frame_indices = list(track.keys())
            best_X = None
            best_dist = -1
            for j in range(len(frame_indices)):
                for k in range(j + 1, len(frame_indices)):
                    f1, f2 = (
                        self._buffer[frame_indices[j]],
                        self._buffer[frame_indices[k]],
                    )
                    pts1 = np.array([track[frame_indices[j]]], dtype=np.float32)
                    pts2 = np.array([track[frame_indices[k]]], dtype=np.float32)

                    X = self._triangulate_two_views(pts1, pts2, f1, f2)
                    mask = self._filter(f1, f2, X)

                    if not mask.any():
                        continue

                    c1 = -f1["R"].T @ f1["t"]
                    c2 = -f2["R"].T @ f2["t"]
                    dist = np.linalg.norm(c1 - c2)

                    if dist > best_dist:
                        best_dist = dist
                        best_X = X[mask][0]

            if best_X is not None:
                self.points3D[i] = best_X

    def _reprojection_residuals(self, X_flat):
        X = X_flat.reshape(-1, 3)
        residuals = []
        for i, track in enumerate(self.tracks):
            if i >= len(X):
                continue
            Xi = X[i]
            for cam_idx, uv in track.items():
                cam = self._buffer[cam_idx]
                K, R, t = cam["K"], cam["R"], cam["t"]
                proj = K @ (R @ Xi.reshape(3, 1) + t)
                u, v = proj[0] / proj[2], proj[1] / proj[2]
                residuals.extend([u - uv[0], v - uv[1]])
        return np.array(residuals).flatten()

    def run_bundle_adjustment(self):
        if not self.points3D:
            raise ValueError("No 3D points to refine. Run initialize_points() first.")

        X0 = np.array(self.points3D).reshape(-1, 3)
        result = least_squares(self._reprojection_residuals, X0.ravel(), verbose=2)
        self.points3D = result.x.reshape(-1, 3).tolist()

        return self.points3D

    def project(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, pts_3d):
        P = K @ np.hstack((R, t))
        pts_h = np.hstack((pts_3d, np.ones((1))))

        pts_2d = (P @ pts_h.T).T
        pts_2d = pts_2d.reshape(-1, 3)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]
        return pts_2d

    def infer(
        self,
        K: np.ndarray,
        frame: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        bboxes: List[Tuple[int]] = None,
        do_ba: bool = False,
    ):
        if do_ba:
            pts_3d = self.run_bundle_adjustment()

            results = {j: [] for j in range(len(bboxes))}
            for pt3d in pts_3d:
                if pt3d is None:
                    continue

                pt2d = self.project(K, R, t, np.array(pt3d)).tolist()[0]
                for j, bbox in enumerate(bboxes):
                    if self._is_in_bbox(pt2d, [bbox]):
                        results[j].append(pt3d)

            return results
        else:
            return None

    def _cheirality(self, f1, f2, X):
        X = X.reshape(-1, 3)
        z1 = (f1["R"] @ X.T + f1["t"]).T[:, 2]
        z2 = (f2["R"] @ X.T + f2["t"]).T[:, 2]
        valid_cheirality = (z1 > 0) & (z2 > 0)
        return valid_cheirality

    def _parallax(self, f1, f2, X):
        c1 = -f1["R"].T @ f1["t"]
        c2 = -f2["R"].T @ f2["t"]
        r1 = X - c1.reshape(1, 3)
        r2 = X - c2.reshape(1, 3)

        cosang = np.sum(r1 * r2, axis=1) / (
            np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12
        )

        angles = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        valid_parallax = angles > 5.0
        return valid_parallax

    def _filter(self, f1, f2, X):
        return self._cheirality(f1, f2, X) & self._parallax(f1, f2, X)

    def __len__(self):
        return len(self._buffer)


def triangulation_postprocess(cv_path, mp4_path, output_path, maxlen):
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_path, exist_ok=True)
    logging.info("Starting Postprocessing...")
    tri_ba_module = TriangulationBAModule(maxlen=maxlen)
    logging.info("Triangulation Bundle Adjustment Module initialized.")

    points3d_json_path = os.path.join(output_path, f"stream_{time_str}_points3d.json")
    firstOutputEntry = True
    with open(points3d_json_path, "w") as f:
        f.write("[\n")

    with open(cv_path, "r") as f:
        cv_data = json.load(f)

    cap = cv2.VideoCapture(mp4_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        logging.error("Could not open video file.")
        return

    frame_idx = 0
    initialized = False
    for frame_idx in tqdm(range(total_frames), desc="Processing frames", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = cv_data[frame_idx]["data"]
        if not frame_data:
            logging.warning(f"No CV data found for frame {frame_idx}")
            continue

        tri_ba_module.insert(
            np.array(frame_data["K"]),
            frame,
            np.array(frame_data["R"]),
            np.array(frame_data["t"]),
            frame_data["bboxes"],
        )

        if initialized and frame_idx % 50 == 0:
            logging.info(f"Inference Triggered at Frame Idx: {frame_idx}")
            points3d = tri_ba_module.infer(
                np.array(frame_data["K"]),
                frame,
                np.array(frame_data["R"]),
                np.array(frame_data["t"]),
                frame_data["bboxes"],
                do_ba=True,
            )

            with open(points3d_json_path, "a") as f:
                if not firstOutputEntry:
                    f.write(",\n")
                json.dump({"frame_idx": frame_idx, "data": points3d}, f, indent=4)
                firstOutputEntry = False

        if len(tri_ba_module) == maxlen and not initialized:
            logging.info("Initializing_3D_points")
            tri_ba_module.initialize_3D_points()
            initialized = True
            points3d = tri_ba_module.infer(
                np.array(frame_data["K"]),
                frame,
                np.array(frame_data["R"]),
                np.array(frame_data["t"]),
                frame_data["bboxes"],
                do_ba=True,
            )

            with open(points3d_json_path, "a") as f:
                if not firstOutputEntry:
                    f.write(",\n")
                json.dump({"frame_idx": frame_idx, "data": points3d}, f, indent=4)
                firstOutputEntry = False

    with open(points3d_json_path, "a") as f:
        f.write("\n]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run triangulation post-processing on video and CV data"
    )
    parser.add_argument(
        "--cv-path",
        type=str,
        required=True,
        help="Path to the CV data JSON file containing camera parameters and detections",
    )
    parser.add_argument(
        "--mp4-path",
        type=str,
        required=True,
        help="Path to the MP4 video file to process",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=120,
        help="Maximum buffer length for triangulation (default: 100)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    args = parser.parse_args()

    try:
        triangulation_postprocess(
            args.cv_path, args.mp4_path, args.output_path, args.maxlen
        )
    except Exception as e:
        logging.error(f"Error during post-processing: {str(e)}")
        raise
