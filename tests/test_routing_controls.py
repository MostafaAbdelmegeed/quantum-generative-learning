import json
from pathlib import Path

import numpy as np
import pytest

from qgen.circuits import ExactCircuit
from qgen.experiment import file_digest, write_json
from qgen.generalization import qiskit_state, target_and_witness
from qgen.routing_controls import audit, deploy, run, sparse_spec


@pytest.fixture
def config():
    return json.loads(
        (Path(__file__).parents[1] / "config/routing-controls-v1.json").read_text()
    )


@pytest.mark.parametrize(
    "condition",
    ["adjacent_pairs", "shuffled_pairs", "noisy_pairs", "parity3", "parity4"],
)
@pytest.mark.parametrize("topology", ["path_bridges", "hub_bridges"])
def test_sparse_targets_are_expressible_in_equal_size_trees(condition, topology):
    spec, witness = sparse_spec(condition, topology)
    assert spec.n_parameters == 11
    assert len(spec.edges) == 5
    seen = {0}
    for _ in range(6):
        seen |= {b for a, b in spec.edges if a in seen} | {
            a for a, b in spec.edges if b in seen
        }
    assert len(seen) == 6
    target, _ = target_and_witness(condition)
    assert np.allclose(ExactCircuit(spec).probabilities(witness), target, atol=1e-12)
    assert np.allclose(qiskit_state(spec, witness).probabilities(), target, atol=1e-12)


@pytest.mark.parametrize("device", ["line", "ring", "star"])
def test_logical_output_order_and_noise(device, config):
    spec, _ = sparse_spec("shuffled_pairs", "hub_bridges")
    theta = np.random.default_rng(72).normal(size=11)
    deployed = deploy(
        spec, theta, device, {**config, "errors": [0, 0.02], "evaluation_shots": 64}, 42
    )
    assert np.allclose(
        deployed["outcomes"][0]["probabilities"],
        ExactCircuit(spec).probabilities(theta),
        atol=1e-12,
    )
    assert deployed["outcomes"][1]["tvd_from_ideal_output"] > 0.001
    assert sum(deployed["outcomes"][1]["sample_counts"]) == 64


def test_archive_replays_routing_and_rejects_compilation_corruption(config, tmp_path):
    config.update(
        conditions=["noisy_pairs"],
        seeds=[42],
        updates=2,
        evaluation_shots=64,
        errors=[0, 0.02],
    )
    path = tmp_path / "config.json"
    write_json(path, config)
    output = tmp_path / "run"
    run(
        path, output, Path(__file__).parents[1] / "results/observable-generalization-v1"
    )
    assert audit(output, replay=True)["noise_evaluations_replayed"] == 12
    result = output / "runs/noisy_pairs-s42.json"
    record = json.loads(result.read_text())
    record["fits"][0]["deployments"][0]["compilation"]["cx_count"] += 1
    write_json(result, record)
    index = json.loads((output / "index.json").read_text())
    index["completed"]["noisy_pairs-s42"] = file_digest(result)
    write_json(output / "index.json", index)
    with pytest.raises(ValueError, match="Compilation replay"):
        audit(output, replay=True)


@pytest.mark.parametrize("topology", ["path_bridges", "hub_bridges"])
def test_optimized_near_identity_rotations_respect_compilation_tolerance(
    config, topology
):
    spec, _ = sparse_spec("shuffled_pairs", topology)
    theta = [
        1.570804233924272,
        -0.0034118841722711466,
        3.1538672979404296,
        -3.141202205626668,
        -1.5704308774346407,
        -0.005645451127096229,
        -0.7810594007775032,
        1.570438354501476,
        0.7850116317141299,
        -3.145893981920516,
        -0.8011534636025066,
    ]
    if topology == "hub_bridges":
        theta = [
            1.57695902246727,
            -3.0418033080818266e-05,
            3.143182855603019,
            -1.5614772824485175,
            -1.5786847115421048,
            1.5843946664306578,
            -1.5707937031028427,
            1.5707662653740972,
            0.7810588037924678,
            -2.3565811074734273,
            -0.7696422615498405,
        ]
    deployed = deploy(
        spec, theta, "line", {**config, "errors": [0], "evaluation_shots": 64}, 3246
    )
    outcome = deployed["outcomes"][0]
    assert outcome["noiseless_max_probability_error"] < 1e-8
    assert (
        np.max(
            np.abs(
                np.array(outcome["probabilities"])
                - ExactCircuit(spec).probabilities(theta)
            )
        )
        < 1e-8
    )
