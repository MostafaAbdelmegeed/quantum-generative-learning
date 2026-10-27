import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from qgen.circuits import CircuitSpec, ExactCircuit, Gate, make_circuit, to_qiskit
from qgen.data import bit_table, make_dataset, target_distribution
from qgen.deployment import evaluate_deployment
from qgen.metrics import evaluate, kernel_matrix, mmd_squared, sampled_mmd
from qgen.training import fit_classical, fit_exact, fit_shots


@pytest.mark.parametrize("family", ["qcbm", "iqp"])
@pytest.mark.parametrize("topology", ["line", "dense", "tree"])
def test_state_and_adjoint_against_independent_references(family, topology):
    spec = make_circuit(4, family, topology, depth=2, seed=17)
    theta = np.random.default_rng(23).normal(size=spec.n_parameters)
    engine = ExactCircuit(spec)
    qc, params = to_qiskit(spec)
    qiskit_state = np.asarray(
        Statevector(qc.assign_parameters(dict(zip(params, theta, strict=True))))
    )
    assert abs(np.vdot(engine.state(theta), qiskit_state)) ** 2 == pytest.approx(
        1, abs=1e-12
    )
    assert np.allclose(engine.probabilities(theta), abs(qiskit_state) ** 2, atol=1e-13)
    target = target_distribution("even_parity", 4)
    kernel = kernel_matrix(4)
    loss, grad = engine.loss_and_grad(theta, target, kernel)
    assert loss == pytest.approx(mmd_squared(abs(qiskit_state) ** 2, target, kernel))
    step = 1e-6
    numerical = []
    for index in range(len(theta)):
        delta = np.zeros_like(theta)
        delta[index] = step
        numerical.append(
            (
                mmd_squared(engine.probabilities(theta + delta), target, kernel)
                - mmd_squared(engine.probabilities(theta - delta), target, kernel)
            )
            / (2 * step)
        )
    assert np.allclose(grad, numerical, atol=1e-8)


def test_iqp_sign_angle_and_endianness_against_matrix_exponential():
    spec = CircuitSpec(
        3, "iqp", "custom", (), (Gate("px", (0, 2), 0), Gate("px", (1,), 1)), 2
    )
    theta = np.array([0.23, -0.41])
    x = np.array([[0, 1], [1, 0]])
    eye = np.eye(2)
    h = theta[0] * np.kron(x, np.kron(eye, x)) + theta[1] * np.kron(
        eye, np.kron(x, eye)
    )
    reference = expm(1j * h)[:, 0]
    qc, parameters = to_qiskit(spec)
    actual = np.asarray(
        Statevector(qc.assign_parameters(dict(zip(parameters, theta, strict=True))))
    )
    assert np.allclose(actual, reference, atol=1e-13)
    assert np.allclose(ExactCircuit(spec).state(theta), reference)


def test_targets_and_independent_splits():
    bas = target_distribution("bars_stripes")
    assert np.count_nonzero(bas) == 10
    images = bit_table(6)[bas > 0].reshape(-1, 2, 3)
    for image in images:
        assert np.all(image == image[:, :1]) or np.all(image == image[:1, :])
    parity = target_distribution("even_parity")
    bits = bit_table(6)
    assert parity @ ((-1.0) ** bits.sum(axis=1)) == pytest.approx(1)
    assert np.allclose(parity @ bits, 0.5)
    data = make_dataset("bars_stripes", 6, 42, 512, 512)
    repeat = make_dataset("bars_stripes", 6, 42, 512, 512)
    assert np.array_equal(data.train_samples, repeat.train_samples)
    assert not np.array_equal(data.train_samples, data.test_samples)


def test_sampled_mmd_matches_explicit_u_statistic():
    kernel = kernel_matrix(2)
    a = np.array([0, 1, 1, 3])
    b = np.array([0, 2, 3, 3, 3])
    p = np.bincount(a, minlength=4) / len(a)
    q = np.bincount(b, minlength=4) / len(b)
    reference = (
        sum(kernel[x, y] for i, x in enumerate(a) for j, y in enumerate(a) if i != j)
        / 12
    )
    reference += (
        sum(kernel[x, y] for i, x in enumerate(b) for j, y in enumerate(b) if i != j)
        / 20
    )
    reference -= 2 * np.mean(kernel[np.ix_(a, b)])
    assert sampled_mmd(p, q, kernel, len(a), len(b)) == pytest.approx(reference)
    p = np.ones(4) / 4
    assert sampled_mmd(p, p, kernel, 4, 4) < 0


def test_metrics_handle_disjoint_support_and_exact_match():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    metrics = evaluate(p, q, q, 1, "test")
    assert metrics["tvd"] == 1
    assert metrics["js_nats"] == pytest.approx(np.log(2))
    assert evaluate(q, q, q, 1, "test")["tvd"] == 0


@pytest.mark.parametrize("family", ["qcbm", "iqp"])
def test_routing_preserves_logical_outputs_and_noise(family):
    spec = make_circuit(4, family, "dense", depth=1)
    theta = np.random.default_rng(47).normal(size=spec.n_parameters)
    metadata, results = evaluate_deployment(spec, theta, [0, 0.02], shots=512)
    expected = ExactCircuit(spec).probabilities(theta)
    assert np.allclose(results[0]["probabilities"], expected, atol=1e-10)
    assert metadata["cx_count"] > 0
    assert np.sum(results[1]["probabilities"]) == pytest.approx(1)
    assert not np.allclose(results[1]["probabilities"], expected)
    assert (
        np.abs(np.asarray(results[0]["sampled_probabilities"]) - expected).sum() / 2
        < 0.15
    )


def test_training_and_budget():
    spec = make_circuit(2, "qcbm", depth=1)
    train = np.array([0.7, 0.1, 0.1, 0.1])
    fit = fit_exact(spec, train, "multiscale", 4, maxiter=60)
    assert fit["training_history"][-1] < fit["training_history"][0] / 10
    shots = fit_shots(spec, train, 10, "fixed", 4, shots=32, updates=3)
    assert shots["resources"]["training_shots"] == 192
    assert shots["resources"]["objective_evaluations"] == 6
    repeated = fit_shots(spec, train, 10, "fixed", 4, shots=32, updates=3)
    assert repeated["parameters"] == shots["parameters"]


def test_classical_autoregressive_distribution_normalizes():
    samples = np.random.default_rng(7).choice([0, 1, 3, 7], size=128)
    for fit in fit_classical(samples, 3, 19):
        p = np.array(fit["probabilities"])
        assert np.all(p >= 0)
        assert p.sum() == pytest.approx(1)
