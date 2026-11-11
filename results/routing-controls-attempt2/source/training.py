"""Exact and finite-shot training, with explicit resource accounting."""

import warnings
from time import perf_counter

import numpy as np
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.neural_network import MLPClassifier

from .circuits import ExactCircuit, to_qiskit
from .data import bit_table
from .metrics import kernel_matrix, sampled_mmd


def initial_parameters(spec, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3107]))
    return rng.uniform(-np.pi, np.pi, spec.n_parameters)


def fit_exact(spec, train, loss_name, seed, maxiter):
    engine = ExactCircuit(spec)
    kernel = kernel_matrix(spec.n, loss_name)
    initial = initial_parameters(spec, seed)
    history = []

    def objective(theta):
        loss, gradient = engine.loss_and_grad(theta, train, kernel)
        history.append(float(loss))
        return loss, gradient

    start = perf_counter()
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
    )
    return {
        "parameters": result.x.tolist(),
        "initial_parameters": initial.tolist(),
        "probabilities": engine.probabilities(result.x).tolist(),
        "initial_probabilities": engine.probabilities(initial).tolist(),
        "training_history": history,
        "training_seconds": perf_counter() - start,
        "optimizer": "L-BFGS-B",
        "converged": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "resources": {
            "objective_evaluations": int(result.nfev),
            "gradient_evaluations": int(result.njev),
            "training_shots": 0,
            "training_state_preparations": 0,
            "exact_diagnostic_evaluations": 2,
        },
    }


def fit_shots(spec, train, train_size, loss_name, seed, shots, updates):
    if min(shots, train_size) < 2 or updates < 1:
        raise ValueError("Invalid sampling or training budget")
    circuit, parameters = to_qiskit(spec)
    circuit.measure_all()
    kernel = kernel_matrix(spec.n, loss_name)
    initial = initial_parameters(spec, seed)
    theta = initial.copy()
    perturb_rng, sampler_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence([seed, 901]).spawn(2)
    ]
    history = []

    def objective(values):
        bound = circuit.assign_parameters(dict(zip(parameters, values, strict=True)))
        sampler = StatevectorSampler(seed=int(sampler_rng.integers(0, 2**31)))
        counts = sampler.run([bound], shots=shots).result()[0].data.meas.get_counts()
        p = np.zeros(1 << spec.n)
        for bitstring, count in counts.items():
            p[int(bitstring, 2)] += count / shots
        return sampled_mmd(p, train, kernel, shots, train_size)

    start = perf_counter()
    for step in range(updates):
        a = 0.15 / (step + 1) ** 0.602
        c = 0.15 / (step + 1) ** 0.101
        direction = perturb_rng.choice([-1.0, 1.0], spec.n_parameters)
        plus = objective(theta + c * direction)
        minus = objective(theta - c * direction)
        gradient = ((plus - minus) / (2 * c)) * direction
        theta -= a * gradient
        history.append(float((plus + minus) / 2))
    seconds = perf_counter() - start
    engine = ExactCircuit(spec)
    return {
        "parameters": theta.tolist(),
        "initial_parameters": initial.tolist(),
        "probabilities": engine.probabilities(theta).tolist(),
        "initial_probabilities": engine.probabilities(initial).tolist(),
        "training_history": history,
        "training_seconds": seconds,
        "optimizer": "SPSA",
        "converged": None,
        "message": "Fixed update budget completed",
        "iterations": updates,
        "resources": {
            "objective_evaluations": 2 * updates,
            "gradient_evaluations": updates,
            "training_shots": 2 * updates * shots,
            "training_state_preparations": 2 * updates * shots,
            "exact_diagnostic_evaluations": 2,
        },
    }


def fit_classical(train_samples, n, seed):
    bits = bit_table(n).astype(float)
    train = bits[train_samples]
    empirical = np.bincount(train_samples, minlength=1 << n) / len(train_samples)
    fitted = []

    def add(name, p, seconds, parameters, info=None):
        fitted.append(
            {
                "model": name,
                "probabilities": p.tolist(),
                "training_seconds": seconds,
                "n_parameters": parameters,
                "fit_info": info or {},
            }
        )

    add("empirical", empirical, 0.0, (1 << n) - 1)
    start = perf_counter()
    means = train.mean(axis=0)
    p = np.prod(np.where(bits, means, 1 - means), axis=1)
    add("independent", p, perf_counter() - start, n)

    start = perf_counter()
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    features = np.column_stack([bits] + [bits[:, a] * bits[:, b] for a, b in pairs])
    moments = empirical @ features

    def objective(weights):
        logits = features @ weights
        log_z = logsumexp(logits)
        probs = np.exp(logits - log_z)
        return float(log_z - moments @ weights + 0.0001 * (weights @ weights) / 2), (
            probs @ features - moments + 0.0001 * weights
        )

    result = minimize(
        objective,
        np.zeros(features.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
    )
    logits = features @ result.x
    p = np.exp(logits - logsumexp(logits))
    add(
        "pairwise",
        p,
        perf_counter() - start,
        len(result.x),
        {
            "converged": bool(result.success),
            "message": str(result.message),
            "nfev": result.nfev,
        },
    )

    start = perf_counter()
    p = np.where(bits[:, 0] == 1, means[0], 1 - means[0])
    parameter_count = 1
    recorded_warnings = []
    for bit in range(1, n):
        y = train[:, bit].astype(int)
        if len(np.unique(y)) < 2:
            conditional = np.full(len(bits), float(y[0]))
            parameter_count += 1
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(8,),
                activation="tanh",
                solver="lbfgs",
                alpha=1.0,
                max_iter=300,
                random_state=seed + bit,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train[:, :bit], y)
            recorded_warnings.extend(str(w.message) for w in caught)
            conditional = model.predict_proba(bits[:, :bit])[
                :, list(model.classes_).index(1)
            ]
            parameter_count += sum(w.size for w in model.coefs_ + model.intercepts_)
        p *= np.where(bits[:, bit] == 1, conditional, 1 - conditional)
    add(
        "autoregressive_mlp",
        p,
        perf_counter() - start,
        parameter_count,
        {"warnings": recorded_warnings, "hidden_units_per_conditional": 8},
    )
    return fitted
