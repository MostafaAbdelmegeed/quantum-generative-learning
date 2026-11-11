import json
from pathlib import Path

import numpy as np
import pytest

from qgen.circuits import ExactCircuit, make_circuit
from qgen.data import bit_table
from qgen.experiment import file_digest, write_json
from qgen.generalization import (
    candidate_masks,
    fit_loglinear,
    fit_observables,
    make_data,
    objective_kernel,
    qiskit_state,
    run,
    score_distribution,
    select_higher,
    target_and_witness,
)
from qgen.generalization_audit import audit
from qgen.iqp import IQPMoments, characters


@pytest.fixture
def config():
    return json.loads(
        (
            Path(__file__).parents[1] / "config/observable-generalization-v1.json"
        ).read_text()
    )


@pytest.mark.parametrize(
    "condition",
    ["adjacent_pairs", "shuffled_pairs", "noisy_pairs", "parity3", "parity4"],
)
def test_constructive_expressibility_and_target_moments(condition):
    target, angles = target_and_witness(condition)
    spec = make_circuit(6, "iqp", "dense")
    assert np.allclose(ExactCircuit(spec).probabilities(angles), target, atol=1e-13)
    assert np.allclose(qiskit_state(spec, angles).probabilities(), target, atol=1e-13)
    assert target.sum() == pytest.approx(1)
    assert np.allclose(target @ bit_table(6), 0.5)
    metrics = score_distribution(target, target, target, condition)
    assert metrics["all_even_probability"] == pytest.approx(
        0.729 if condition == "noisy_pairs" else 1
    )
    assert metrics["block_moment_mae"] == 0
    if condition in {"parity3", "parity4"}:
        assert np.allclose(
            target @ characters(bit_table(6), candidate_masks(6, 2, 2)), 0
        )


def test_paired_dataset_permutation_and_independent_splits():
    adjacent = make_data("adjacent_pairs", 42, 512, 512)
    shuffled = make_data("shuffled_pairs", 42, 512, 512)
    bits = bit_table(6)
    assert np.array_equal(
        bits[adjacent["train"]][:, [0, 2, 4, 1, 3, 5]], bits[shuffled["train"]]
    )
    assert adjacent["train"] != adjacent["heldout"]


def test_selector_finds_higher_constraints_and_rejects_redundancy():
    for condition, expected in [
        ("parity3", {(0, 2, 4), (1, 3, 5)}),
        ("parity4", {(0, 2, 3, 5)}),
    ]:
        target, _ = target_and_witness(condition)
        train = bit_table(6)[target > 0].astype(np.int32)
        selected, details = select_higher(train)
        supports = {tuple(np.flatnonzero(row)) for row in selected}
        assert expected <= supports
        codes = selected @ (1 << np.arange(6))
        assert (
            len(
                {
                    int(
                        np.bitwise_xor.reduce(
                            codes[np.array(choice, dtype=bool)], initial=0
                        )
                    )
                    for choice in bit_table(3)
                }
            )
            == 8
        )
        assert details["character_evaluations"] == len(train) * 50


def test_selected_gradient_against_adjoint_and_finite_difference():
    table = bit_table(6).astype(np.int32)
    train = table[make_data("parity3", 17, 128, 128)["train"]]
    masks, _ = select_higher(train)
    spec = make_circuit(6, "iqp", "dense")
    theta = np.random.default_rng(312).normal(size=21)
    empirical = np.bincount(train @ (1 << np.arange(6)), minlength=64) / len(train)
    kernel = objective_kernel(6, masks, 0.5)
    engine = ExactCircuit(spec)
    _, grad = engine.loss_and_grad(theta, empirical, kernel)
    moments, jacobian = IQPMoments(spec).moments_and_jacobian(theta, masks, table)
    _, base_grad = engine.loss_and_grad(theta, empirical, objective_kernel(6, None, 0))
    residual = moments - characters(train, masks).mean(axis=0)
    assert np.allclose(
        grad, 0.5 * base_grad + residual @ jacobian / len(masks), atol=1e-12
    )
    for j in range(21):
        delta = np.eye(21)[j] * 1e-6
        finite = (
            engine.loss_and_grad(theta + delta, empirical, kernel)[0]
            - engine.loss_and_grad(theta - delta, empirical, kernel)[0]
        ) / 2e-6
        assert grad[j] == pytest.approx(finite, abs=1e-8)


def test_enumerated_control_shares_masks_and_adam_schedule(config, monkeypatch):
    config = {**config, "updates": 4}
    spec = make_circuit(6, "iqp", "dense")
    train = bit_table(6)[make_data("parity3", 42, 128, 128)["train"]].astype(np.int32)
    masks, _ = select_higher(train)
    original = IQPMoments.loss_and_grad

    def exact_halves(self, theta, masks, train, left, right):
        table = bit_table(6).astype(np.int32)
        return original(self, theta, masks, train, table, table)

    monkeypatch.setattr(IQPMoments, "loss_and_grad", exact_halves)
    mc = fit_observables(spec, train, 42, config, masks, "mc")
    exact = fit_observables(spec, train, 42, config, masks, "enumerated")
    assert np.allclose(mc["parameters"], exact["parameters"], atol=1e-13)
    assert np.allclose(mc["training_history"], exact["training_history"], atol=1e-13)
    assert (
        mc["resources"]["classical_phase_evaluations"]
        == 2 * exact["resources"]["classical_phase_evaluations"]
    )


def test_classical_higher_order_control_can_learn_parity(config):
    target, _ = target_and_witness("parity4")
    train = bit_table(6)[target > 0].astype(np.int32)
    fit = fit_loglinear(train, candidate_masks(6, 1, 4), config, "order4_maxent")
    assert fit["fit_info"]["converged"]
    assert np.abs(np.array(fit["probabilities"]) - target).sum() / 2 < 0.001


def test_archive_audit_replays_and_rejects_resigned_metric_corruption(config, tmp_path):
    config.update(conditions=["parity4"], seeds=[42], updates=2)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    output = tmp_path / "series"
    run(config_path, output)
    assert audit(output, replay=True)["adam_replays"] == 6
    path = output / "runs/parity4-s42.json"
    record = json.loads(path.read_text())
    record["fits"][0]["metrics"]["tvd"] += 0.1
    write_json(path, record)
    index = json.loads((output / "index.json").read_text())
    index["completed"]["parity4-s42"] = file_digest(path)
    write_json(output / "index.json", index)
    with pytest.raises(ValueError, match="tvd"):
        audit(output)
    with pytest.raises(FileExistsError):
        run(config_path, output)
