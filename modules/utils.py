import math
import time
from typing import Callable, Tuple

import cv2
import numpy as np
import psutil
from skimage.metrics import structural_similarity as ssim

from config import N_LOGICAL
import os


def clear_terminal():
    if os.name == "nt":
        _ = os.system("cls")
    else:
        _ = os.system("clear")


def measure_cpu_usage(
    proc: psutil.Process, fn: Callable, *args, **kwargs
) -> Tuple[any, float, float, float, float]:
    """
    Measures the CPU usage of a function.

    Args:
        proc: The process to measure.
        fn: The function to measure.
        *args: The arguments to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        A tuple containing the output of the function, the time taken, the total CPU percentage,
        the machine CPU percentage, and the core equivalent.
    """
    proc.cpu_percent(interval=None)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    pct_all = proc.cpu_percent(interval=None)
    dt = time.perf_counter() - t0

    core_equiv = pct_all / 100.0
    pct_machine = pct_all / N_LOGICAL

    return out, dt, pct_all, pct_machine, core_equiv


def img_preprocess(image: np.ndarray, dim: Tuple[int, int] = (64, 36)) -> np.ndarray:
    """
    Preprocesses an image by resizing and converting it to grayscale.

    Args:
        image: The image to preprocess.
        dim: The dimensions to resize the image to.

    Returns:
        The preprocessed image.
    """
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA) if dim else image
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return grayscale


def mean_abs_diff(curr: np.ndarray, ref: np.ndarray) -> float:
    """
    Calculates the mean absolute difference between two images.

    Args:
        curr: The current image.
        ref: The reference image.

    Returns:
        The mean absolute difference.
    """
    if curr.shape != ref.shape:
        raise ValueError(
            f"Current array does not match reference array in shape. "
            f"Expected {ref.shape} from current array but got {curr.shape}"
        )
    c = curr.astype(np.float32) / 255.0
    r = ref.astype(np.float32) / 255.0
    return float(np.mean(np.abs(c - r)))


def hist_diff(curr: np.ndarray, ref: np.ndarray) -> float:
    """
    Calculates the histogram difference between two images.

    Args:
        curr: The current image.
        ref: The reference image.

    Returns:
        The histogram difference.
    """
    cur_hist = cv2.calcHist([curr], [0], None, [64], [0, 255])
    ref_hist = cv2.calcHist([ref], [0], None, [64], [0, 255])
    cv2.normalize(cur_hist, cur_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    cv2.normalize(ref_hist, ref_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(cur_hist, ref_hist, cv2.HISTCMP_CHISQR))


def ssim_diff(curr: np.ndarray, ref: np.ndarray) -> float:
    """
    Calculates the structural similarity index between two images.

    Args:
        curr: The current image.
        ref: The reference image.

    Returns:
        The structural similarity index.
    """
    c = curr.astype(np.float32) / 255.0
    r = ref.astype(np.float32) / 255.0
    dr = float(max(1e-6, c.max() - c.min(), r.max() - r.min()))
    score, _ = ssim(c, r, full=True, data_range=dr)
    return float(score)


def get_camera_intrinsic(height: float, width: float, fov: float) -> np.ndarray:
    """
    Calculates the camera intrinsic matrix.

    Args:
        height: The height of the image.
        width: The width of the image.
        fov: The field of view of the camera.

    Returns:
        The camera intrinsic matrix.
    """
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    fx = width / (2 * math.tan(fov / 2))
    fy = fx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def get_camera_extrinsic(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Calculates the camera extrinsic matrix.

    Args:
        rotation: The rotation quaternion.
        translation: The translation vector.

    Returns:
        The camera extrinsic matrix.
    """
    if rotation.shape != (4,):
        raise ValueError("Rotation quaternion must have shape (4,).")
    if translation.shape != (3,):
        raise ValueError("Translation vector must have shape (3,).")

    l2_norm = np.linalg.norm(rotation)
    if l2_norm == 0:
        raise ValueError("Rotation quaternion has zero length.")
    rotation /= l2_norm

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
