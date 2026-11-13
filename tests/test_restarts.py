import json
from pathlib import Path

import numpy as np
import pytest

from qgen.circuits import ExactCircuit, make_circuit
from qgen.data import bit_table
from qgen.diagnostics import fit_full, without_times
from qgen.experiment import file_digest, write_json
from qgen.generalization import make_data, qiskit_state
from qgen.restarts import (
    audit,
    initialization_vectors,
    marginal_product_angles,
    run,
    select_training_index,
    train_policy,
    validate_config,
)
from qgen.training import initial_parameters

ROOT = Path(__file__).parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "config/initialization-restarts-v1.json").read_text())


def test_product_initialization_represents_training_marginals():
    spec = make_circuit(6, "iqp", "dense")
    means = np.array([0, 0.1, 0.3, 0.6, 0.9, 1])
    theta = marginal_product_angles(spec, means)
    expected = np.prod(np.where(bit_table(6), means, 1 - means), axis=1)
    assert np.allclose(ExactCircuit(spec).probabilities(theta), expected, atol=1e-14)
    assert np.allclose(qiskit_state(spec, theta).probabilities(), expected, atol=1e-14)


def test_paired_initializations_retain_original_and_share_normal_draws(config):
    spec = make_circuit(6, "iqp", "dense")
    means = np.linspace(0.1, 0.9, 6)
    first = initialization_vectors(spec, means, 42, config)
    assert first == initialization_vectors(spec, means, 42, config)
    assert first["uniform"][0] == initial_parameters(spec, 42).tolist()
    assert len({tuple(v) for v in first["uniform"]}) == 4
    nominal = marginal_product_angles(spec, means)
    for near, product in zip(
        first["near_identity"], first["marginal_product"], strict=True
    ):
        assert np.allclose(np.array(near) / 0.1, (np.array(product) - nominal) / 0.05)


@pytest.mark.parametrize("values", [[], [np.nan], [1, np.inf], [[1, 2]]])
def test_selection_rejects_invalid_scores(values):
    with pytest.raises(ValueError, match="finite training losses"):
        select_training_index(values)


def test_selection_uses_training_scores_and_first_exact_tie():
    assert select_training_index([0.3, 0.1, 0.1, 0.2]) == 1


def test_explicit_initialization_preserves_old_path_and_input(config):
    spec = make_circuit(6, "iqp", "dense")
    train = bit_table(6)[make_data("parity4", 42, 512, 1024)["train"]].astype(np.int32)
    config = {**config, "updates": 3}
    initial = initial_parameters(spec, 42)
    before = initial.copy()
    original = fit_full(spec, train, 42, config)
    explicit = fit_full(spec, train, 42, config, initial=initial)
    assert without_times(original) == without_times(explicit)
    assert np.array_equal(initial, before)
    for invalid in [np.zeros(20), np.full(21, np.nan)]:
        with pytest.raises(ValueError):
            fit_full(spec, train, 42, config, initial=invalid)


def test_policy_selection_precedes_evaluation_and_counts_diagnostics(config):
    spec = make_circuit(6, "iqp", "dense")
    train = bit_table(6)[make_data("parity3", 42, 512, 1024)["train"]].astype(np.int32)
    config = {
        **config,
        "total_updates": 4,
        "allocations": {"single": [4], "restarts": [1] * 4},
    }
    initials = initialization_vectors(spec, train.mean(axis=0), 42, config)["uniform"]
    policies = [
        train_policy(spec, train, 42, config, None, initials, a)
        for a in config["allocations"]
    ]
    for policy in policies:
        assert policy["resources"]["gradient_updates"] == 4
        assert policy["resources"]["training_phase_evaluations"] == 4 * 4096
        assert all("metrics" not in c for c in policy["candidates"])
        assert policy["selected_candidate"] == np.argmin(
            [c["diagnostics"]["final"]["loss"] for c in policy["candidates"]]
        )
    assert (
        policies[1]["resources"]["diagnostic_phase_evaluations"]
        - policies[0]["resources"]["diagnostic_phase_evaluations"]
        == 24576
    )


def test_unequal_allocations_rejected(config):
    config["allocations"]["restarts"] = [400] * 3
    with pytest.raises(ValueError, match="budgets must match"):
        validate_config(config)


@pytest.mark.parametrize("corruption", ["selection", "resources", "initialization"])
def test_archive_replay_and_semantic_corruption_rejection(config, tmp_path, corruption):
    config.update(
        conditions=["parity4"],
        seeds=[42],
        total_updates=4,
        allocations={"single": [4], "restarts": [1] * 4},
    )
    path = tmp_path / "config.json"
    write_json(path, config)
    output = tmp_path / "run"
    run(path, output, ROOT / "results/observable-diagnostics-v1")
    report = audit(output, replay=True)
    assert report["training_replays"] == report["candidates_checked"] == 60
    assert report["policies_checked"] == 24
    result_path = output / "runs/parity4-s42.json"
    record = json.loads(result_path.read_text())
    original = record["initializations"]["marginal_product"][0][0]
    record["initializations"]["marginal_product"][0][0] = float(
        np.nextafter(original, np.inf)
    )
    write_json(result_path, record)
    index = json.loads((output / "index.json").read_text())
    index["completed"]["parity4-s42"] = file_digest(result_path)
    write_json(output / "index.json", index)
    assert audit(output)["passed"]
    policy = next(p for p in record["policies"] if p["allocation"] == "restarts")
    if corruption == "selection":
        policy["selected_candidate"] = (policy["selected_candidate"] + 1) % 4
        message = "Training-only selection"
    elif corruption == "resources":
        policy["resources"]["gradient_updates"] += 1
        message = "Policy resource"
    else:
        record["initializations"]["marginal_product"][0][0] += 1e-3
        message = "initializations"
    write_json(result_path, record)
    index = json.loads((output / "index.json").read_text())
    index["completed"]["parity4-s42"] = file_digest(result_path)
    write_json(output / "index.json", index)
    with pytest.raises(ValueError, match=message):
        audit(output)
