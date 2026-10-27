import numpy as np
import pytest

from qgen.circuits import ExactCircuit, make_circuit
from qgen.data import bit_table
from qgen.iqp import IQPMoments, characters, strong_pair_masks
from qgen.metrics import kernel_matrix
from qgen.scaling import paired_bits


def test_pair_selection_uses_correlations_after_column_permutation():
    original = paired_bits(6, 512, np.random.default_rng(42))
    order = np.array([0, 2, 4, 1, 3, 5])
    selected = strong_pair_masks(original[:, order], 3)
    assert {tuple(np.flatnonzero(row)) for row in selected} == {(0, 3), (1, 4), (2, 5)}
    with pytest.raises(ValueError):
        strong_pair_masks(original, 16)


def test_mixture_kernel_gradient_matches_full_probability_oracle():
    bits = bit_table(4).astype(np.int32)
    samples = paired_bits(4, 128, np.random.default_rng(31))
    priority = strong_pair_masks(samples, 2)
    basis = characters(bits, priority)
    kernel = 0.5 * kernel_matrix(4) + 0.5 * basis @ basis.T / len(priority)
    assert np.linalg.eigvalsh(kernel).min() > -1e-12
    spec = make_circuit(4, "iqp", "line")
    theta = np.random.default_rng(1).normal(size=spec.n_parameters)
    empirical = np.bincount(samples @ (1 << np.arange(4)), minlength=16) / len(samples)
    actual_loss, actual_grad = ExactCircuit(spec).loss_and_grad(
        theta, empirical, kernel
    )
    base_loss, base_grad = ExactCircuit(spec).loss_and_grad(
        theta, empirical, kernel_matrix(4)
    )
    moments, jac = IQPMoments(spec).moments_and_jacobian(theta, priority, bits)
    residual = moments - characters(samples, priority).mean(axis=0)
    expected_loss = 0.5 * base_loss + 0.5 * np.mean(residual**2)
    expected_grad = 0.5 * base_grad + residual @ jac / len(priority)
    assert actual_loss == pytest.approx(expected_loss, abs=1e-12)
    assert np.allclose(actual_grad, expected_grad, atol=1e-12)
