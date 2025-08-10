import numpy as np
import math


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
