import numpy as np
import cv2
from typing import Tuple, List
from collections import deque
from scipy.optimize import least_squares


class TriangulationModule:
    def __init__(self, k: np.ndarray):
        self.k = k
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __call__(self, frame1, R1, t1, frame2, R2, t2, bboxes):
        return self.infer(frame1, R1, t1, frame2, R2, t2, bboxes)

    def _feature_match(
        self, frame1: np.ndarray, frame2: np.ndarray, bboxes: List[Tuple[int]]
    ):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

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

        valid_pts1 = []
        valid_pts2 = []
        for pt1, pt2 in zip(pts1, pts2):
            if self._is_in_bbox(pt1, bboxes) and self._is_in_bbox(pt2, bboxes):
                valid_pts1.append(pt1)
                valid_pts2.append(pt2)

        pts1 = np.array(valid_pts1, dtype=np.float32)
        pts2 = np.array(valid_pts2, dtype=np.float32)

        return pts1, pts2

    def infer(self, frame1, R1, t1, frame2, R2, t2, bboxes):
        P1 = self.k @ np.hstack((R1, t1))
        P2 = self.k @ np.hstack((R2, t2))

        pts1, pts2 = self._feature_match(frame1, frame2, bboxes)
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

    def update(self, k: np.ndarray):
        self.k = k

    def _is_in_bbox(self, pts: Tuple[float, float], bboxes: List[Tuple[int]]):
        x, y = pts
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True

        return False


class TriangulationNewModule:
    def __init__(self, maxlen: int = 100):
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
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, w), min(y2, h)

            if x2 > x1 and y2 > y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 255
        return mask

    def insert_buffer(
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

        E, mask = cv2.findEssentialMat(
            pts1, pts2, f1["K"], method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

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
                self.tracks.append(new_track)

                self.points3D.append(None)

                f1, f2 = self._buffer[idx1], self._buffer[idx2]
                pts1 = np.array([[u1, v1]], dtype=np.float32)
                pts2 = np.array([[u2, v2]], dtype=np.float32)
                X = self._triangulate_two_views(pts1, pts2, f1, f2)[0]

                self.points3D[-1] = X

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

                    X = self._triangulate_two_views(pts1, pts2, f1, f2)[0]

                    c1 = -f1["R"].T @ f1["t"]
                    c2 = -f2["R"].T @ f2["t"]
                    dist = np.linalg.norm(c1 - c2)

                    if dist > best_dist:
                        best_dist = dist
                        best_X = X

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
        return np.array(residuals)

    def run_bundle_adjustment(self):
        if not self.points3D:
            raise ValueError("No 3D points to refine. Run initialize_points() first.")
        X0 = np.array(self.points3D).reshape(-1, 3)
        result = least_squares(self._reprojection_residuals, X0.ravel(), verbose=2)
        refined = result.x.reshape(-1, 3)
        self.points3D = refined
        return refined

    def project(self, K: np.ndarray, R: np.ndarray, t: np.ndarray):
        P = K @ np.hstack((R, t))
        pts_3d = np.array(self.points3D)
        pts_h = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))

        pts_2d = (P @ pts_h.T).T
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
        self.insert_buffer(K, frame, R, t, bboxes)
        if do_ba:
            pts_3d = self.run_bundle_adjustment()
            pts_2d = self.project(K, R, t)

            results = {j: [] for j in range(len(bboxes))}
            for pt2d, pt3d in zip(pts_2d, pts_3d):
                for j, bbox in enumerate(bboxes):
                    if self._is_in_bbox(pt2d, [bbox]):
                        results[j].append(pt3d)

            return results
        else:
            return None
