"""Geometric utility functions for angle computation."""

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle in degrees between two 2D or 3D vectors.

    Args:
        v1: First vector (array-like, any length ≥ 2).
        v2: Second vector (same length as v1).

    Returns:
        Angle in degrees in [0, 180].
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at joint b formed by the chain a–b–c, in degrees.

    Args:
        a: Point proximal to b (array-like, 2D or 3D).
        b: The joint being measured.
        c: Point distal from b.

    Returns:
        Angle at b in degrees in [0, 180].
    """
    return angle_between_vectors(
        np.asarray(a) - np.asarray(b),
        np.asarray(c) - np.asarray(b),
    )
