"""Enumerated synthetic distributions; columns follow Qiskit's qubit order."""

from dataclasses import dataclass
from itertools import product

import numpy as np


def bit_table(n: int) -> np.ndarray:
    if not 1 <= n <= 12:
        raise ValueError("Enumerated experiments support 1 to 12 qubits")
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def probabilities_from_samples(samples: np.ndarray, n: int) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 1 or not len(samples):
        raise ValueError("Expected a nonempty vector of integer outcomes")
    if not np.issubdtype(samples.dtype, np.integer) or np.any(samples < 0) or np.any(samples >= 1 << n):
        raise ValueError("Sample outside the circuit support")
    return np.bincount(samples, minlength=1 << n) / len(samples)


def target_distribution(name: str, n: int = 6) -> np.ndarray:
    bits = bit_table(n)
    if name == "bars_stripes":
        if n != 6:
            raise ValueError("The frozen Bars-and-Stripes dataset is 2 by 3")
        patterns = set()
        for rows in product([0, 1], repeat=2):
            pixels = np.repeat(rows, 3)
            patterns.add(int(pixels @ (1 << np.arange(n))))
        for cols in product([0, 1], repeat=3):
            pixels = np.tile(cols, 2)
            patterns.add(int(pixels @ (1 << np.arange(n))))
        p = np.zeros(1 << n)
        p[list(patterns)] = 1
    elif name == "gaussian_mixture":
        x = np.arange(1 << n)
        scale = (1 << n) / 64
        p = 0.4 * np.exp(-0.5 * ((x - 18 * scale) / (4 * scale)) ** 2) / 4
        p += 0.6 * np.exp(-0.5 * ((x - 45 * scale) / (5 * scale)) ** 2) / 5
    elif name == "even_parity":
        p = (bits.sum(axis=1) % 2 == 0).astype(float)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return p / p.sum()


@dataclass
class Dataset:
    name: str
    n: int
    target: np.ndarray
    train_samples: np.ndarray
    test_samples: np.ndarray

    @property
    def train(self):
        return probabilities_from_samples(self.train_samples, self.n)

    @property
    def test(self):
        return probabilities_from_samples(self.test_samples, self.n)


def make_dataset(name: str, n: int, seed: int, train_size: int, test_size: int) -> Dataset:
    if min(train_size, test_size) < 2:
        raise ValueError("Both splits need at least two samples")
    target = target_distribution(name, n)
    train_rng, test_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    return Dataset(name, n, target, train_rng.choice(len(target), train_size, p=target),
                   test_rng.choice(len(target), test_size, p=target))

