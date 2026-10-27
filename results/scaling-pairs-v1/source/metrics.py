"""Losses and diagnostics on a small, fully enumerable binary support."""

import numpy as np
from scipy.special import xlogy

from .data import bit_table


def kernel_matrix(n: int, loss: str = "multiscale") -> np.ndarray:
    bits = bit_table(n)
    distance = np.count_nonzero(bits[:, None, :] != bits[None, :, :], axis=-1)
    if loss == "fixed":
        bandwidths = [1.0]
    elif loss == "multiscale":
        bandwidths = n * np.array([0.01, 0.1, 0.25, 0.5, 1, 10])
    else:
        raise ValueError(f"Unknown loss: {loss}")
    return np.mean([np.exp(-distance / (2 * b)) for b in bandwidths], axis=0)


def mmd_squared(p, q, kernel):
    difference = np.asarray(p) - np.asarray(q)
    return float(difference @ kernel @ difference)


def sampled_mmd(p, q, kernel, shots, train_size):
    if min(shots, train_size) < 2:
        raise ValueError("U statistics require at least two samples")
    diagonal = np.diag(kernel)
    pp = (shots * (p @ kernel @ p) - p @ diagonal) / (shots - 1)
    qq = (train_size * (q @ kernel @ q) - q @ diagonal) / (train_size - 1)
    return float(pp + qq - 2 * p @ kernel @ q)


def validate_distribution(p):
    p = np.asarray(p)
    if (
        p.ndim != 1
        or not np.all(np.isfinite(p))
        or np.any(p < -1e-12)
        or not np.isclose(p.sum(), 1, atol=1e-9)
    ):
        raise ValueError("Invalid probability distribution")


def evaluate(p, target, held_out, n, dataset_name):
    for distribution in [p, target, held_out]:
        validate_distribution(distribution)
    p = np.maximum(p, 0)
    midpoint = (p + target) / 2
    safe = np.where(midpoint > 0, midpoint, 1)
    js = 0.5 * np.sum(xlogy(p, p / safe) + xlogy(target, target / safe))
    parity = 1 - 2 * (bit_table(n).sum(axis=1) % 2)
    result = {
        "tvd": float(np.abs(p - target).sum() / 2),
        "js_nats": float(js),
        "heldout_mmd": mmd_squared(p, held_out, kernel_matrix(n)),
        "parity_error": float(abs((p - target) @ parity)),
    }
    if dataset_name == "bars_stripes":
        support = target > 0
        result["invalid_mass"] = float(p[~support].sum())
        result["expected_coverage_1024"] = float(np.mean(1 - (1 - p[support]) ** 1024))
    return result
