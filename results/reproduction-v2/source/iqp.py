"""Classical IQP moment estimation without statevectors or circuit samples.

Convention: U(theta) = product_j exp(+i theta_j X_Sj).
See Recio-Armengol and Bowles, arXiv:2501.04776, Eq. (8).
"""

from time import perf_counter

import numpy as np

from .training import initial_parameters


def binary_matrix(values, width, name):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != width or len(values) == 0:
        raise ValueError(f"{name} must be a nonempty matrix with {width} columns")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{name} must contain only zero and one")
    return values.astype(np.int32)


def characters(samples, masks):
    """Walsh characters, one row per sample and one column per mask."""
    return 1.0 - 2.0 * ((samples @ masks.T) % 2)


def bandwidths(n, loss):
    if loss == "fixed":
        return np.array([1.0])
    if loss == "multiscale":
        return n * np.array([.01, .1, .25, .5, 1, 10])
    raise ValueError(f"Unknown loss: {loss}")


def sample_masks(n, count, loss, rng):
    """Fourier measure of exp(-Hamming/(2*b)); includes the identity mask."""
    if n < 1 or count < 1:
        raise ValueError("Positive numbers of bits and masks are required")
    b = rng.choice(bandwidths(n, loss), size=count)
    inclusion = -np.expm1(-1 / (2 * b)) / 2
    return (rng.random((count, n)) < inclusion[:, None]).astype(np.int32)


def mask_weights(masks, loss):
    """Exact Fourier weights, intended for enumerated small-system tests."""
    masks = np.asarray(masks)
    n = masks.shape[1]
    weight = masks.sum(axis=1)
    inclusion = -np.expm1(-1 / (2 * bandwidths(n, loss))) / 2
    return np.mean(inclusion[:, None] ** weight *
                   (1 - inclusion[:, None]) ** (n - weight), axis=0)


class IQPMoments:
    """Memory scales with the sampled masks and bitstrings, never with 2**n."""

    def __init__(self, spec):
        if spec.family != "iqp" or len(spec.gates) != spec.n_parameters:
            raise ValueError("Expected an IQP circuit with one parameter per generator")
        self.spec = spec
        self.supports = np.zeros((spec.n_parameters, spec.n), dtype=np.int32)
        seen = set()
        for gate in spec.gates:
            if (gate.kind != "px" or gate.parameter is None or
                    not 0 <= gate.parameter < spec.n_parameters or gate.parameter in seen or
                    not gate.qubits or len(set(gate.qubits)) != len(gate.qubits) or
                    any(q < 0 or q >= spec.n for q in gate.qubits)):
                raise ValueError("Malformed IQP generator")
            seen.add(gate.parameter)
            self.supports[gate.parameter, list(gate.qubits)] = 1

    def moments_and_jacobian(self, theta, masks, uniform_bits):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.spec.n_parameters,) or not np.all(np.isfinite(theta)):
            raise ValueError("Invalid parameter vector")
        masks = binary_matrix(masks, self.spec.n, "masks")
        z = binary_matrix(uniform_bits, self.spec.n, "uniform_bits")
        active = (masks @ self.supports.T) % 2
        signs = characters(z, self.supports)
        phase = (2 * active * theta) @ signs.T
        moments = np.cos(phase).mean(axis=1)
        jacobian = -2 * active * (np.sin(phase) @ signs) / len(z)
        return moments, jacobian

    def loss_and_grad(self, theta, masks, train_bits, z_left, z_right):
        """Unbiased loss/gradient against the fixed empirical training target.

        z_left and z_right must be independently sampled uniform bitstrings.
        Reusing them would add a parameter-dependent Monte Carlo variance term.
        The estimate can be negative; its expectation is MMD squared.
        """
        masks = binary_matrix(masks, self.spec.n, "masks")
        train = binary_matrix(train_bits, self.spec.n, "train_bits")
        data_moments = characters(train, masks).mean(axis=0)
        left, left_grad = self.moments_and_jacobian(theta, masks, z_left)
        right, right_grad = self.moments_and_jacobian(theta, masks, z_right)
        left -= data_moments
        right -= data_moments
        loss = float(np.mean(left * right))
        gradient = np.mean(right[:, None] * left_grad + left[:, None] * right_grad, axis=0)
        return loss, gradient


def fit_iqp_mc(spec, train_bits, loss_name, seed, updates, masks_per_step,
               samples_per_half, learning_rate):
    """Adam with fresh classical samples; final parameters, no test selection."""
    if min(updates, masks_per_step, samples_per_half) < 1 or not learning_rate > 0:
        raise ValueError("All training budgets and learning rate must be positive")
    train_bits = binary_matrix(train_bits, spec.n, "train_bits")
    engine = IQPMoments(spec)
    initial = initial_parameters(spec, seed)
    theta = initial.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5201]))
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    history = []
    start = perf_counter()
    for step in range(1, updates + 1):
        masks = sample_masks(spec.n, masks_per_step, loss_name, rng)
        left = rng.integers(0, 2, (samples_per_half, spec.n), dtype=np.int32)
        right = rng.integers(0, 2, (samples_per_half, spec.n), dtype=np.int32)
        loss, gradient = engine.loss_and_grad(theta, masks, train_bits, left, right)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Nonfinite IQP objective or gradient")
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient ** 2
        corrected_first = first / (1 - .9 ** step)
        corrected_second = second / (1 - .999 ** step)
        fraction = (step - 1) / max(1, updates - 1)
        rate = learning_rate * (.1 + .9 * (1 + np.cos(np.pi * fraction)) / 2)
        theta -= rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        history.append(loss)
    return {
        "parameters": theta.tolist(), "initial_parameters": initial.tolist(),
        "training_history": history, "training_seconds": perf_counter() - start,
        "optimizer": "Adam", "converged": None, "message": "Fixed update budget completed",
        "iterations": updates,
        "estimator": "Independent paired estimates against the empirical training target",
        "resources": {
            "objective_evaluations": updates, "gradient_evaluations": updates,
            "training_shots": 0, "training_state_preparations": 0,
            "training_statevector_evaluations": 0, "exact_diagnostic_evaluations": 0,
            "classical_uniform_bitstrings": 2 * updates * samples_per_half,
            "classical_fourier_masks": updates * masks_per_step,
            "classical_phase_evaluations": 2 * updates * masks_per_step * samples_per_half,
        },
    }
