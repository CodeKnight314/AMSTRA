import numpy as np
import cv2
from typing import Tuple, List


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
