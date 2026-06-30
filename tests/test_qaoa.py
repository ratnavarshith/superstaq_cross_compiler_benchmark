"""Tests for make_qaoa and graph_edge_count."""
import pytest
from qiskit import QuantumCircuit
from scxb.qaoa import make_qaoa, graph_edge_count


# --- Edge count invariants (exact for deterministic topologies) ---

@pytest.mark.parametrize("n", [4, 6, 8])
def test_line_graph_has_n_minus_1_edges(n):
    assert graph_edge_count(n, "line") == n - 1


@pytest.mark.parametrize("n", [4, 6, 8])
def test_ring_graph_has_n_edges(n):
    assert graph_edge_count(n, "ring") == n


@pytest.mark.parametrize("n", [4, 5, 6])
def test_complete_graph_has_correct_edge_count(n):
    assert graph_edge_count(n, "complete") == n * (n - 1) // 2


def test_random_regular_has_correct_edge_count():
    # n=6, degree=3: 6*3/2 = 9 edges
    assert graph_edge_count(6, "random_regular", degree=3) == 9


# --- Circuit structure ---

@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("graph", ["line", "ring", "complete", "er"])
def test_qaoa_returns_quantum_circuit_with_correct_qubit_count(n, graph):
    qc = make_qaoa(n, graph=graph)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == n


def test_qaoa_random_regular():
    qc = make_qaoa(6, graph="random_regular", degree=3)
    assert qc.num_qubits == 6


def test_qaoa_has_no_measurements():
    for graph in ("line", "ring", "complete", "er"):
        qc = make_qaoa(4, graph=graph)
        assert qc.num_clbits == 0, f"{graph}: unexpected classical bits"


def test_qaoa_rzz_count_matches_edge_count():
    # p=1: one RZZ per edge per layer
    n = 5
    for graph, expected in [("line", n - 1), ("ring", n), ("complete", n * (n - 1) // 2)]:
        qc = make_qaoa(n, graph=graph, p=1)
        assert qc.count_ops().get("rzz", 0) == expected, (
            f"{graph}: expected {expected} RZZ gates"
        )


def test_qaoa_rzz_count_scales_with_p():
    n = 4
    for p in (1, 2, 3):
        qc = make_qaoa(n, graph="ring", p=p)
        assert qc.count_ops().get("rzz", 0) == n * p


# --- Error handling ---

def test_qaoa_random_regular_odd_product_raises():
    # n=5, degree=3: 5*3=15 is odd
    with pytest.raises(ValueError, match="must be even"):
        make_qaoa(5, graph="random_regular", degree=3)


def test_qaoa_unknown_graph_raises():
    with pytest.raises(ValueError, match="Unknown graph family"):
        make_qaoa(4, graph="hypercube")


# --- Reproducibility ---

def test_qaoa_same_seed_same_gate_counts():
    qc1 = make_qaoa(5, graph="er", seed=42)
    qc2 = make_qaoa(5, graph="er", seed=42)
    assert qc1.count_ops() == qc2.count_ops()
