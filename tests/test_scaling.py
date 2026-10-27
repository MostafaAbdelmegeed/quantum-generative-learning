import numpy as np
import pytest

from qgen.circuits import ExactCircuit, make_circuit
from qgen.data import bit_table
from qgen.metrics import kernel_matrix, sampled_mmd
from qgen.scaling import (
    decode_bits,
    encode_bits,
    pair_diagnostics,
    paired_bits,
    sample_mmd,
    sample_qiskit_mps,
)


def test_scaling_data_and_string_order():
    bits = paired_bits(32, 100, np.random.default_rng(23))
    assert pair_diagnostics(bits)["valid_fraction"] == 1
    assert pair_diagnostics(bits)["pair_agreement"] == 1
    assert np.array_equal(bits, decode_bits(encode_bits(bits), 32))
    assert encode_bits(np.array([[1, 0, 0, 0]])) == ["0001"]
    assert (
        pair_diagnostics(np.array([[0, 1, 0, 0], [1, 1, 1, 1]]))["valid_fraction"]
        == 0.5
    )
    with pytest.raises(ValueError):
        paired_bits(3, 100, np.random.default_rng(1))


def test_chunked_sample_mmd_against_full_kernel_u_statistic():
    bits = bit_table(4)
    a, b = np.array([0, 0, 3, 7, 12]), np.array([1, 2, 6, 6, 9, 15])
    p, q = np.bincount(a, minlength=16) / len(a), np.bincount(b, minlength=16) / len(b)
    expected = sampled_mmd(p, q, kernel_matrix(4), len(a), len(b))
    assert sample_mmd(bits[a], bits[b], block_size=2) == pytest.approx(
        expected, abs=1e-13
    )


def test_matrix_product_sampling_matches_exact_and_replays():
    spec = make_circuit(4, "iqp", "line")
    theta = np.random.default_rng(27).normal(size=spec.n_parameters)
    bits, metadata = sample_qiskit_mps(spec, theta, 2048, 45)
    sampled = np.bincount(bits @ (1 << np.arange(4)), minlength=16) / len(bits)
    exact = ExactCircuit(spec).probabilities(theta)
    assert np.abs(sampled - exact).sum() / 2 < 0.08
    repeat, _ = sample_qiskit_mps(spec, theta, 2048, 45)
    assert np.array_equal(bits, repeat)
    assert metadata["truncation_threshold"] == 0
