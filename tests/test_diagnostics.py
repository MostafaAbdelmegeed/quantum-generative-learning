import json
from pathlib import Path

import numpy as np
import pytest

from qgen.circuits import ExactCircuit, make_circuit
from qgen.data import bit_table, probabilities_from_samples
from qgen.diagnostics import FullObjective, audit, fit_full, run, select_order_penalized
from qgen.experiment import file_digest, write_json
from qgen.generalization import make_data, objective_kernel, target_and_witness


@pytest.fixture
def config():
    return json.loads(
        (
            Path(__file__).parents[1] / "config/observable-diagnostics-v1.json"
        ).read_text()
    )


@pytest.mark.parametrize(
    "condition", ["adjacent_pairs", "shuffled_pairs", "parity3", "parity4"]
)
def test_order_penalty_prefers_simple_true_constraints(condition, config):
    target, _ = target_and_witness(condition)
    train = bit_table(6)[target > 0].astype(np.int32)
    selected = select_order_penalized(train, config)
    orders = np.sum(selected["masks"], axis=1)
    if condition.endswith("pairs"):
        assert list(orders) == [2, 2, 2]
    else:
        degree = int(condition[-1])
        assert list(orders[: 2 if degree == 3 else 1]) == (
            [3, 3] if degree == 3 else [4]
        )
    assert np.allclose(
        selected["penalized_scores"],
        np.array(selected["scores"]) / (np.sum(selected["candidates"], axis=1) - 1),
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_full_objective_matches_probability_adjoint(weighted, config):
    spec = make_circuit(6, "iqp", "dense")
    data = make_data("parity4", 42, 512, 1024)
    train = bit_table(6)[data["train"]].astype(np.int32)
    priority = (
        np.array(select_order_penalized(train, config)["masks"]) if weighted else None
    )
    objective = FullObjective(spec, train, priority)
    theta = np.random.default_rng(89).normal(size=21)
    loss, grad = objective(theta)
    empirical = probabilities_from_samples(np.array(data["train"]), 6)
    expected_loss, expected_grad = ExactCircuit(spec).loss_and_grad(
        theta, empirical, objective_kernel(6, priority, 0.5)
    )
    assert loss == pytest.approx(expected_loss, abs=1e-12)
    assert np.allclose(grad, expected_grad, atol=1e-12)
    assert objective.weights.sum() == pytest.approx(1)
    assert np.all(objective.weights > 0)


def test_full_adam_determinism_and_budget(config):
    spec = make_circuit(6, "iqp", "dense")
    train = bit_table(6)[make_data("parity3", 42, 512, 1024)["train"]].astype(np.int32)
    config = {**config, "updates": 4}
    first = fit_full(spec, train, 42, config)
    second = fit_full(spec, train, 42, config)
    assert first["parameters"] == second["parameters"]
    assert first["training_history"] == second["training_history"]
    assert first["resources"]["sampled_masks"] == 0
    assert first["resources"]["classical_phase_evaluations"] == 4 * 4096


def test_diagnostic_archive_and_corruption_rejection(config, tmp_path):
    config.update(conditions=["parity4"], seeds=[42], update_budgets=[2])
    path = tmp_path / "config.json"
    write_json(path, config)
    output = tmp_path / "run"
    reference = Path(__file__).parents[1] / "results/observable-generalization-v1"
    run(path, output, reference)
    assert audit(output, replay=True)["adam_replays"] == 12
    result_path = output / "runs/parity4-s42.json"
    record = json.loads(result_path.read_text())
    record["fits"][0]["resources"]["classical_phase_evaluations"] += 1
    write_json(result_path, record)
    index = json.loads((output / "index.json").read_text())
    index["completed"]["parity4-s42"] = file_digest(result_path)
    write_json(output / "index.json", index)
    with pytest.raises(ValueError, match="Phase budget"):
        audit(output)
