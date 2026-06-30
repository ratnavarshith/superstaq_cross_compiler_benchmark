"""Tests for make_clifford."""
import pytest
from qiskit import QuantumCircuit
from scxb.clifford import make_clifford


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clifford_qubit_count(n):
    qc = make_clifford(n)
    assert qc.num_qubits == n


def test_clifford_has_no_measurements():
    qc = make_clifford(4)
    assert qc.num_clbits == 0


def test_clifford_returns_quantum_circuit():
    qc = make_clifford(4)
    assert isinstance(qc, QuantumCircuit)


def test_clifford_full_random_same_seed_same_gate_counts():
    qc1 = make_clifford(4, seed=0)
    qc2 = make_clifford(4, seed=0)
    assert qc1.count_ops() == qc2.count_ops()


# --- Depth-controlled mode ---

def test_clifford_depth_returns_quantum_circuit():
    qc = make_clifford(4, depth=10)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4


def test_clifford_depth_gate_count_equals_depth():
    # Each iteration appends exactly one gate, so total ops == depth.
    depth = 20
    qc = make_clifford(4, depth=depth)
    assert sum(qc.count_ops().values()) == depth


def test_clifford_depth_has_no_measurements():
    qc = make_clifford(4, depth=10)
    assert qc.num_clbits == 0


def test_clifford_depth_same_seed_same_gate_counts():
    qc1 = make_clifford(4, depth=15, seed=42)
    qc2 = make_clifford(4, depth=15, seed=42)
    assert qc1.count_ops() == qc2.count_ops()


def test_clifford_depth_exact_count_matches_depth():
    for depth in (5, 10, 30):
        qc = make_clifford(3, depth=depth, seed=0)
        assert sum(qc.count_ops().values()) == depth


def test_clifford_single_qubit_depth_never_produces_cx():
    # n=1: no valid CX pairs; every gate should fall back to H.
    qc = make_clifford(1, depth=10, seed=0)
    assert "cx" not in qc.count_ops()
    assert sum(qc.count_ops().values()) == 10
