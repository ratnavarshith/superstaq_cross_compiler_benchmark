"""Tests for make_qft."""
import pytest
from qiskit import QuantumCircuit
from scxb.qft import make_qft


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_qft_has_correct_qubit_count(n):
    qc = make_qft(n)
    assert qc.num_qubits == n


def test_qft_has_no_measurements():
    qc = make_qft(4)
    assert qc.num_clbits == 0


def test_qft_decomposed_has_multiple_elementary_gates():
    # The decomposed QFT (copied from the BlueprintCircuit's internal data)
    # should have many gates (h, cp, swap), not a single composite block.
    qc = make_qft(4, decomposed=True)
    assert sum(qc.count_ops().values()) > 1


def test_qft_undecomposed_is_single_block():
    # With decomposed=False the QFT is wrapped as one composite gate
    # so the outer circuit has exactly one instruction.
    qc = make_qft(4, decomposed=False)
    assert sum(qc.count_ops().values()) == 1


def test_qft_approximation_reduces_total_gate_count():
    # Dropping the smallest controlled-phase rotations must reduce gate count.
    qc_full = make_qft(6, approximation_degree=0, decomposed=True)
    qc_approx = make_qft(6, approximation_degree=3, decomposed=True)
    total_full = sum(qc_full.count_ops().values())
    total_approx = sum(qc_approx.count_ops().values())
    assert total_approx < total_full


def test_qft_returns_quantum_circuit():
    qc = make_qft(4)
    assert isinstance(qc, QuantumCircuit)
