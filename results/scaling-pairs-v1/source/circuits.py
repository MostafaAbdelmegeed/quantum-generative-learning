"""Small-state training circuits and independently executable Qiskit exports."""

from dataclasses import asdict, dataclass
from itertools import combinations, pairwise

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


@dataclass(frozen=True)
class Gate:
    kind: str
    qubits: tuple[int, ...]
    parameter: int | None = None


@dataclass(frozen=True)
class CircuitSpec:
    n: int
    family: str
    topology: str
    edges: tuple[tuple[int, int], ...]
    gates: tuple[Gate, ...]
    n_parameters: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["n"],
            data["family"],
            data["topology"],
            tuple(tuple(e) for e in data["edges"]),
            tuple(
                Gate(g["kind"], tuple(g["qubits"]), g["parameter"])
                for g in data["gates"]
            ),
            data["n_parameters"],
        )


def topology_edges(n: int, topology: str, seed: int) -> tuple[tuple[int, int], ...]:
    if topology == "line":
        return tuple((i, i + 1) for i in range(n - 1))
    if topology == "dense":
        return tuple(combinations(range(n), 2))
    if topology == "tree":
        if n < 2:
            return ()
        code = np.random.default_rng(seed).integers(0, n, n - 2)
        degree = np.ones(n, dtype=int) + np.bincount(code, minlength=n)
        edges = []
        for parent in code:
            leaf = int(np.flatnonzero(degree == 1)[0])
            edges.append(tuple(sorted((leaf, int(parent)))))
            degree[leaf] -= 1
            degree[parent] -= 1
        edges.append(tuple(np.flatnonzero(degree == 1).tolist()))
        return tuple(sorted(edges))
    raise ValueError(f"Unknown topology: {topology}")


def make_circuit(
    n: int, family: str, topology: str = "line", depth: int = 3, seed: int = 0
) -> CircuitSpec:
    if not 1 <= n <= 1024 or depth < 1:
        raise ValueError("Invalid circuit size or depth")
    edges = topology_edges(n, topology, seed)
    gates = []
    parameter = 0
    if family == "qcbm":
        for _ in range(depth):
            for q in range(n):
                for kind in ["ry", "rz"]:
                    gates.append(Gate(kind, (q,), parameter))
                    parameter += 1
            gates.extend(Gate("cx", edge) for edge in edges)
        for q in range(n):
            gates.append(Gate("ry", (q,), parameter))
            parameter += 1
    elif family == "iqp":
        for support in [(q,) for q in range(n)] + list(edges):
            gates.append(Gate("px", support, parameter))
            parameter += 1
    else:
        raise ValueError(f"Unknown circuit family: {family}")
    return CircuitSpec(n, family, topology, edges, tuple(gates), parameter)


def to_qiskit(spec: CircuitSpec) -> tuple[QuantumCircuit, ParameterVector]:
    theta = ParameterVector("theta", spec.n_parameters)
    circuit = QuantumCircuit(spec.n)
    if spec.family == "iqp":
        # H exp(i theta Z_S) H = exp(i theta X_S), with all diagonal terms commuting.
        circuit.h(range(spec.n))
        for gate in spec.gates:
            support = gate.qubits
            for a, b in pairwise(support):
                circuit.cx(a, b)
            circuit.rz(-2 * theta[gate.parameter], support[-1])
            for a, b in reversed(list(pairwise(support))):
                circuit.cx(a, b)
        circuit.h(range(spec.n))
    else:
        for gate in spec.gates:
            if gate.kind == "cx":
                circuit.cx(*gate.qubits)
            else:
                getattr(circuit, gate.kind)(theta[gate.parameter], gate.qubits[0])
    return circuit, theta


class ExactCircuit:
    """NumPy adjoint engine, cross-checked against Qiskit for every circuit family."""

    def __init__(self, spec: CircuitSpec):
        if not 1 <= spec.n <= 12:
            raise ValueError("Exact state diagnostics are limited to 12 qubits")
        self.spec = spec
        self.indices = np.arange(1 << spec.n)
        self.zero = np.zeros(1 << spec.n, dtype=complex)
        self.zero[0] = 1
        self.zero_indices = {
            q: self.indices[(self.indices & (1 << q)) == 0] for q in range(spec.n)
        }

    def apply(self, state, gate, angle=0.0, derivative=False, inverse=False):
        if inverse:
            angle = -angle
        if gate.kind == "cx":
            control, target = gate.qubits
            perm = self.indices ^ (((self.indices >> control) & 1) << target)
            return state[perm]
        if gate.kind == "px":
            mask = sum(1 << q for q in gate.qubits)
            flipped = state[self.indices ^ mask]
            if derivative:
                return -np.sin(angle) * state + 1j * np.cos(angle) * flipped
            return np.cos(angle) * state + 1j * np.sin(angle) * flipped
        q = gate.qubits[0]
        z = self.zero_indices[q]
        o = z | (1 << q)
        result = np.empty_like(state)
        if gate.kind == "ry":
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            if derivative:
                result[z] = (-s * state[z] - c * state[o]) / 2
                result[o] = (c * state[z] - s * state[o]) / 2
            else:
                result[z] = c * state[z] - s * state[o]
                result[o] = s * state[z] + c * state[o]
        elif gate.kind == "rz":
            result[z] = np.exp(-0.5j * angle) * state[z]
            result[o] = np.exp(0.5j * angle) * state[o]
            if derivative:
                result[z] *= -0.5j
                result[o] *= 0.5j
        else:
            raise ValueError(f"Unknown gate: {gate.kind}")
        return result

    def state(self, theta):
        if np.shape(theta) != (self.spec.n_parameters,):
            raise ValueError("Wrong parameter shape")
        state = self.zero.copy()
        for gate in self.spec.gates:
            state = self.apply(
                state, gate, 0 if gate.parameter is None else theta[gate.parameter]
            )
        return state

    def probabilities(self, theta):
        return np.abs(self.state(theta)) ** 2

    def loss_and_grad(self, theta, target, kernel):
        history = [self.zero.copy()]
        for gate in self.spec.gates:
            angle = 0 if gate.parameter is None else theta[gate.parameter]
            history.append(self.apply(history[-1], gate, angle))
        state = history[-1]
        diff = np.abs(state) ** 2 - target
        kd = kernel @ diff
        loss = float(diff @ kd)
        adjoint = 2 * kd * state
        gradient = np.zeros(self.spec.n_parameters)
        for gate, before in zip(
            reversed(self.spec.gates), reversed(history[:-1]), strict=True
        ):
            angle = 0 if gate.parameter is None else theta[gate.parameter]
            if gate.parameter is not None:
                derivative = self.apply(before, gate, angle, derivative=True)
                gradient[gate.parameter] += 2 * np.vdot(adjoint, derivative).real
            adjoint = self.apply(adjoint, gate, angle, inverse=True)
        return loss, gradient
