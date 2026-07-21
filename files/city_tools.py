"""Gauge and novelty helpers for the reconstructed local runner."""

from __future__ import annotations

import numpy as np


def compute_gauge_fixed(predictions: np.ndarray, gauge_matrices: np.ndarray) -> np.ndarray:
    """Apply per-timestamp gauge matrices with explicit shape validation."""

    pred = np.asarray(predictions, dtype=np.float64)
    gauges = np.asarray(gauge_matrices, dtype=np.float64)
    if pred.ndim != 2:
        raise ValueError(f"predictions must be 2D, got {pred.shape}")
    if gauges.ndim != 3:
        raise ValueError(f"gauge_matrices must be 3D, got {gauges.shape}")
    if gauges.shape[0] != pred.shape[0] or gauges.shape[2] != pred.shape[1]:
        raise ValueError(
            "gauge/prediction shape mismatch: "
            f"gauges={gauges.shape}, predictions={pred.shape}"
        )
    raw = np.einsum("tij,tj->ti", gauges, pred)
    raw -= raw.mean(axis=1, keepdims=True)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def make_gauge_matrices(n_rows: int, n_assets: int, seed: int = 314159) -> np.ndarray:
    """Generate deterministic orthonormal cross-sectional gauge matrices."""

    rng = np.random.default_rng(seed)
    output = np.empty((n_rows, n_assets - 1, n_assets), dtype=np.float64)
    ones = np.ones((n_assets, 1), dtype=np.float64) / np.sqrt(n_assets)
    for t in range(n_rows):
        matrix = rng.standard_normal((n_assets, n_assets - 1))
        matrix -= ones @ (ones.T @ matrix)
        q, _ = np.linalg.qr(matrix)
        output[t] = q[:, : n_assets - 1].T
    return output


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return 90.0
    left = left[mask]
    right = right[mask]
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-15:
        return 90.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cosine))))


def nearest_city(signal: np.ndarray, city_count: int = 12, seed: int = 271828) -> tuple[float, str]:
    """Return deterministic synthetic novelty angle and nearest city name."""

    flat = np.asarray(signal, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    best_angle = 90.0
    best_name = "local_city_0000"
    for i in range(city_count):
        city = rng.standard_normal(flat.shape[0])
        angle = angle_degrees(flat, city)
        if angle < best_angle:
            best_angle = angle
            best_name = f"local_city_{i:04d}"
    return best_angle, best_name
