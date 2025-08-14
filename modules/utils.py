import numpy as np
import math
import cv2
from typing import Tuple
from skimage.metrics import structural_similarity as ssim
import psutil
import time
from config import N_LOGICAL


def measure_cpu_usage(proc: psutil.Process, fn, *args, **kwargs):
    proc.cpu_percent(interval=None)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    pct_all = proc.cpu_percent(interval=None)
    dt = time.perf_counter() - t0

    core_equiv = pct_all / 100.0
    pct_machine = pct_all / N_LOGICAL

    return out, dt, pct_all, pct_machine, core_equiv


def img_preprocess(image: np.array, dim: Tuple[int, int] = (64, 36)) -> np.ndarray:
    if dim is not None:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        resized = image

    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return grayscale


def feature_match(kp1, des1, kp2, des2, K):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return pts1, pts2


def nomralize(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    return np.hstack([x, y], axis=1)


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


def get_camera_intrinsic(height: float, width: float, fov: float):
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    fx = width / (2 * math.tan(fov / 2))
    fy = fx
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return matrix


def get_camera_extrinsic(rotation: np.array, translation: np.array):
    if rotation.shape != (4,):
        raise ValueError("Rotation quaternion must have shape (4,).")
    if translation.shape != (3,):
        raise ValueError("Translation vector must have shape (3,).")

    l2_norm = np.linalg.norm(rotation)
    if l2_norm == 0:
        raise ValueError("Rotation quaternion has zero length.")
    rotation = rotation / l2_norm

    x, y, z, w = rotation
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def get_object_wc(
    coord: np.array,
    depth: np.array,
    intrinsic_matrix: np.array,
    extrinsic_matrix: np.array,
):
    N = coord.shape[0]
    if coord.shape != (N, 2):
        raise ValueError()
    if depth.shape != (N, 1):
        raise ValueError()
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError()
    if extrinsic_matrix.shape != (4, 4):
        raise ValueError()

    homog_coord = np.hstack([coord, np.ones((coord.shape[0], 1))])
    camera_coords = (np.linalg.inv(intrinsic_matrix) @ homog_coord.T) * depth.T
    world_coords = extrinsic_matrix @ np.vstack(
        [camera_coords, np.ones((1, coord.shape[0]))]
    )
    world_coords = world_coords[:3, :].T
    return world_coords
