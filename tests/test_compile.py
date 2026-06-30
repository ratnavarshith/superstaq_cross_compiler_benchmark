"""Tests for compile_qiskit and CompileResult.

Uses AerSimulator as backend in all tests because:
  - always available without extra installs
  - no coupling map → no SWAP insertion (routing_overhead == 1.0 for circuits
    with 2Q gates, or 0.0 for circuits with none)
  - no noise model → fidelity is always -1.0 (which we test explicitly)

The important invariants we pin down:
  - CompileResult fields match the spec's name, family, n_qubits
  - two_q_before matches what we'd count from the circuit directly
  - compile_s > 0
  - fidelity == -1.0 for Aer (no noise model comparison makes sense)
  - fidelity == -1.0 for BV regardless of backend (has measurements)
  - fidelity == -1.0 for n > fidelity_max_n
"""
import pytest
from qiskit_aer import AerSimulator

from scxb.compile import CompileResult, compile_qiskit
from scxb.spec import make_circuit


@pytest.fixture(scope="module")
def aer():
    return AerSimulator()


@pytest.fixture
def qaoa_ring_spec():
    return make_circuit("qaoa", n=4, graph="ring", p=1)


@pytest.fixture
def qft_spec():
    return make_circuit("qft", n=4)


@pytest.fixture
def bv_spec():
    return make_circuit("bv", n=4)


@pytest.fixture
def clifford_spec():
    return make_circuit("clifford", n=3)


# --- CompileResult structure ---

def test_compile_qiskit_returns_compile_result(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer, opt_level=1, layout_method="sabre")
    assert isinstance(r, CompileResult)


def test_compile_result_spec_fields_match(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer)
    assert r.spec_name == qaoa_ring_spec.name
    assert r.family == "qaoa"
    assert r.n_qubits == 4


def test_compile_result_compiler_is_qiskit(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer)
    assert r.compiler == "qiskit"


def test_compile_result_opt_level_matches(aer, qaoa_ring_spec):
    for opt in (0, 1, 2, 3):
        r = compile_qiskit(qaoa_ring_spec, aer, opt_level=opt)
        assert r.opt_level == opt


def test_compile_result_layout_method_matches(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer, layout_method="sabre")
    assert r.layout_method == "sabre"


def test_compile_result_seed_matches(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer, seed=7)
    assert r.seed == 7


# --- Pre-compilation metrics ---

def test_two_q_before_matches_ring_edge_count(aer, qaoa_ring_spec):
    # Ring n=4: 4 edges × 1 RZZ per layer × p=1 = 4 two-qubit gates
    r = compile_qiskit(qaoa_ring_spec, aer)
    assert r.two_q_before == 4


def test_depth_before_is_positive(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer)
    assert r.depth_before > 0


# --- Post-compilation metrics ---

def test_compile_s_is_positive(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer)
    assert r.compile_s > 0.0


def test_two_q_after_is_non_negative(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer)
    assert r.two_q_after >= 0


def test_total_gates_is_positive(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer)
    assert r.total_gates > 0


def test_routing_overhead_zero_when_no_original_2q_gates(aer):
    # Complete-0-edge circuit: make a BV with secret=0 (no oracle CX).
    # BV with n=2 has 1 input qubit; secret=0 → 0 oracle CX.
    spec = make_circuit("bv", n=2, secret=0)
    r = compile_qiskit(spec, aer)
    assert r.two_q_before == 0
    assert r.routing_overhead == 0.0


# --- Fidelity rules ---

def test_fidelity_is_minus_one_for_aer_backend(aer, qaoa_ring_spec):
    # Bare AerSimulator has no noise model, so fidelity comparison is meaningless.
    r = compile_qiskit(qaoa_ring_spec, aer, measure_fidelity=True, fidelity_max_n=10)
    assert r.fidelity == -1.0


def test_fidelity_is_minus_one_for_bv(aer, bv_spec):
    # BV has measurements; fidelity must be skipped regardless of backend.
    r = compile_qiskit(bv_spec, aer, measure_fidelity=True, fidelity_max_n=10)
    assert r.fidelity == -1.0


def test_fidelity_is_minus_one_when_n_exceeds_max(aer, qaoa_ring_spec):
    # qaoa_ring n=4; fidelity_max_n=3 → skipped
    r = compile_qiskit(qaoa_ring_spec, aer, measure_fidelity=True, fidelity_max_n=3)
    assert r.fidelity == -1.0


def test_fidelity_is_minus_one_when_disabled(aer, qaoa_ring_spec):
    r = compile_qiskit(qaoa_ring_spec, aer, measure_fidelity=False)
    assert r.fidelity == -1.0


# --- All circuit families compile without errors ---

@pytest.mark.parametrize("family,kwargs", [
    ("qaoa", {"graph": "line", "p": 1}),
    ("qft", {"approximation_degree": 0}),
    ("bv", {"secret": "all_ones"}),
    ("clifford", {}),
])
def test_all_families_compile_successfully(aer, family, kwargs):
    spec = make_circuit(family, n=4, **kwargs)
    r = compile_qiskit(spec, aer, opt_level=1, layout_method="sabre")
    assert isinstance(r, CompileResult)
    assert r.n_qubits == 4
