"""Qiskit routing and local noise evaluation in logical output order."""

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from .circuits import to_qiskit


def compile_circuit(spec, theta):
    circuit, parameters = to_qiskit(spec)
    bound = circuit.assign_parameters(dict(zip(parameters, theta, strict=True)))
    bound.measure_all()
    coupling = CouplingMap.from_line(spec.n, bidirectional=True)
    return transpile(
        bound,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=coupling,
        optimization_level=3,
        seed_transpiler=17,
    )


def logical_probabilities(physical, measured_circuit):
    mapping = {}
    for instruction in measured_circuit.data:
        if instruction.operation.name == "measure":
            q = measured_circuit.find_bit(instruction.qubits[0]).index
            c = measured_circuit.find_bit(instruction.clbits[0]).index
            mapping[q] = c
    if set(mapping.values()) != set(range(measured_circuit.num_clbits)):
        raise ValueError("Expected complete one-to-one output measurement")
    indices = np.arange(len(physical))
    logical = np.zeros(len(physical), dtype=int)
    for physical_bit, classical_bit in mapping.items():
        logical |= ((indices >> physical_bit) & 1) << classical_bit
    result = np.zeros(1 << measured_circuit.num_clbits)
    np.add.at(result, logical, physical)
    return result


def noise_model(error):
    if not 0 <= error <= 1:
        raise ValueError("Depolarizing parameter must be in [0,1]")
    noise = NoiseModel()
    if error:
        noise.add_all_qubit_quantum_error(depolarizing_error(error, 2), ["cx"])
        noise.add_all_qubit_quantum_error(
            depolarizing_error(error / 10, 1), ["sx", "x"]
        )
    return noise


def evaluate_deployment(spec, theta, errors, shots=4096, seed=17):
    compiled = compile_circuit(spec, theta)
    metadata = {
        "depth": compiled.depth(),
        "cx_count": compiled.count_ops().get("cx", 0),
        "gate_counts": dict(compiled.count_ops()),
        "n_physical_qubits": compiled.num_qubits,
        "target": "bidirectional_line",
        "optimization_level": 3,
        "transpiler_seed": 17,
        "qasm": None,
    }
    from qiskit import qasm3

    metadata["qasm"] = qasm3.dumps(compiled)
    results = []
    for error in errors:
        simulator = AerSimulator(
            method="density_matrix",
            noise_model=noise_model(error),
            max_parallel_threads=1,
        )
        probe = compiled.remove_final_measurements(inplace=False)
        probe.save_probabilities()
        raw = (
            simulator.run(probe, shots=1, seed_simulator=seed)
            .result()
            .data(0)["probabilities"]
        )
        distribution = logical_probabilities(np.asarray(raw), compiled)
        counts = (
            simulator.run(compiled, shots=shots, seed_simulator=seed)
            .result()
            .get_counts()
        )
        sampled = np.zeros(1 << spec.n)
        for key, count in counts.items():
            sampled[int(key.replace(" ", ""), 2)] += count / shots
        results.append(
            {
                "cx_depolarizing": error,
                "probabilities": distribution.tolist(),
                "sampled_probabilities": sampled.tolist(),
                "evaluation_shots": shots,
                "sampler_seed": seed,
            }
        )
    return metadata, results
