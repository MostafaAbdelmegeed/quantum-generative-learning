from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qgen.circuits import CircuitSpec, ExactCircuit, Gate, make_circuit, to_qiskit
from qgen.data import bit_table
from qgen.iqp import IQPMoments, characters, fit_iqp_mc, mask_weights, sample_masks
from qgen.metrics import kernel_matrix


@pytest.mark.parametrize("topology", ["line", "dense", "tree"])
def test_moments_and_gradient_against_qiskit(topology):
    n = 4
    spec = make_circuit(n, "iqp", topology, seed=137)
    theta = np.random.default_rng(91).normal(size=spec.n_parameters)
    bits = bit_table(n).astype(np.int32)
    engine = IQPMoments(spec)
    moments, jac = engine.moments_and_jacobian(theta, bits, bits)
    circuit, params = to_qiskit(spec)

    def reference(values):
        p = Statevector(
            circuit.assign_parameters(dict(zip(params, values, strict=True)))
        ).probabilities()
        return p @ characters(bits, bits)

    assert np.allclose(moments, reference(theta), atol=1e-12, rtol=0)
    for j in range(spec.n_parameters):
        delta = np.eye(spec.n_parameters)[j] * 1e-6
        difference = (reference(theta + delta) - reference(theta - delta)) / 2e-6
        assert np.allclose(jac[:, j], difference, atol=1e-8, rtol=0)


def test_three_body_generator_and_identity_observable():
    spec = CircuitSpec(
        3, "iqp", "custom", (), (Gate("px", (0, 1, 2), 0), Gate("px", (1,), 1)), 2
    )
    bits = bit_table(3).astype(np.int32)
    theta = np.array([0.31, -0.19])
    moments, grad = IQPMoments(spec).moments_and_jacobian(theta, bits, bits)
    p = ExactCircuit(spec).probabilities(theta)
    assert np.allclose(moments, p @ characters(bits, bits), atol=1e-12)
    assert moments[0] == 1
    assert np.all(grad[0] == 0)


@pytest.mark.parametrize("loss", ["fixed", "multiscale"])
def test_fourier_kernel_and_exact_objective_gradient(loss):
    n = 4
    bits = bit_table(n).astype(np.int32)
    weights = mask_weights(bits, loss)
    basis = characters(bits, bits)
    kernel = kernel_matrix(n, loss)
    assert weights.sum() == pytest.approx(1)
    assert np.allclose((basis * weights) @ basis.T, kernel, atol=1e-13)
    spec = make_circuit(n, "iqp", "dense")
    theta = np.random.default_rng(17).normal(size=spec.n_parameters)
    train = np.array([0, 0, 2, 5, 7, 8, 11, 15])
    empirical = np.bincount(train, minlength=1 << n) / len(train)
    expected_loss, expected_grad = ExactCircuit(spec).loss_and_grad(
        theta, empirical, kernel
    )
    moments, jac = IQPMoments(spec).moments_and_jacobian(theta, bits, bits)
    residual = moments - characters(bits[train], bits).mean(axis=0)
    assert weights @ residual**2 == pytest.approx(expected_loss, abs=1e-12)
    assert np.allclose(2 * (weights * residual) @ jac, expected_grad, atol=1e-12)


def test_independent_pair_estimator_is_unbiased_by_exhaustive_average():
    # Average over every possible pair of one-sample draws and every Fourier mask.
    n = 2
    bits = bit_table(n).astype(np.int32)
    spec = make_circuit(n, "iqp", "dense")
    theta = np.array([0.4, -0.6, 0.7])
    train = bits[[0, 0, 1, 2, 3]]
    target = np.array([0.4, 0.2, 0.2, 0.2])
    reference_loss, reference_grad = ExactCircuit(spec).loss_and_grad(
        theta, target, kernel_matrix(n)
    )
    engine = IQPMoments(spec)
    loss, grad = 0.0, np.zeros_like(theta)
    for mask, weight in zip(bits, mask_weights(bits, "multiscale"), strict=True):
        for left, right in product(bits, repeat=2):
            estimate, derivative = engine.loss_and_grad(
                theta, mask[None], train, left[None], right[None]
            )
            loss += weight * estimate / len(bits) ** 2
            grad += weight * derivative / len(bits) ** 2
    assert loss == pytest.approx(reference_loss, abs=1e-13)
    assert np.allclose(grad, reference_grad, atol=1e-13)


def test_mask_sampling_matches_kernel_measure():
    rng = np.random.default_rng(42)
    masks = sample_masks(3, 100_000, "multiscale", rng)
    counts = np.bincount(masks @ (1 << np.arange(3)), minlength=8) / len(masks)
    weights = mask_weights(bit_table(3), "multiscale")
    assert np.all(
        np.abs(counts - weights) < 6 * np.sqrt(weights * (1 - weights) / len(masks))
    )


def test_mc_training_is_deterministic_and_uses_no_statevector(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Statevectors must not enter classical IQP training")

    monkeypatch.setattr(ExactCircuit, "__init__", forbidden)
    monkeypatch.setattr(Statevector, "__init__", forbidden)
    spec = make_circuit(32, "iqp", "line")
    train = np.random.default_rng(9).integers(0, 2, (64, 32))
    fit = fit_iqp_mc(spec, train, "multiscale", 4, 4, 8, 16, 0.03)
    repeated = fit_iqp_mc(spec, train, "multiscale", 4, 4, 8, 16, 0.03)
    assert fit["parameters"] == repeated["parameters"]
    assert fit["training_history"] == repeated["training_history"]
    assert fit["resources"]["training_shots"] == 0
    assert fit["resources"]["training_statevector_evaluations"] == 0
    assert fit["resources"]["classical_uniform_bitstrings"] == 128
    assert fit["resources"]["classical_phase_evaluations"] == 1024
    assert not np.array_equal(fit["parameters"], fit["initial_parameters"])


def test_invalid_data_and_unsafe_exact_size_fail_before_allocation():
    with pytest.raises(ValueError, match="12 qubits"):
        ExactCircuit(make_circuit(32, "iqp"))
    spec = make_circuit(2, "iqp")
    with pytest.raises(ValueError, match="zero and one"):
        fit_iqp_mc(spec, [[0, 2]], "fixed", 0, 2, 2, 2, 0.03)
