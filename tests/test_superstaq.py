"""Tests for compile_superstaq and run_superstaq_sweep.

All tests mock qiskit_superstaq.SuperstaqProvider so no live API key or
network access is required. The mock's cq_compile() returns a real
QuantumCircuit (not a MagicMock) so _count_2q, count_ops(), and depth()
work normally on the compiled output.

Live integration tests are gated behind RUN_SUPERSTAQ_LIVE=1 and skipped
by default so CI never hits the network or consumes API credits.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from qiskit import QuantumCircuit

# Skip the entire file if qiskit_superstaq is not installed.
# @patch("qiskit_superstaq.X") resolves the import at decoration time,
# so without the package every test fails at mock setup rather than being skipped.
pytest.importorskip("qiskit_superstaq")

from scxb.compile import CompileResult, compile_superstaq
from scxb.spec import make_circuit
from scxb.sweep import run_superstaq_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_output(compiled: QuantumCircuit) -> MagicMock:
    """Build a mock CompilerOutput whose .circuit is a real QuantumCircuit."""
    out = MagicMock()
    out.circuit = compiled
    return out


def _compiled_cz(n: int = 4) -> QuantumCircuit:
    """A plausible post-SuperstaQ circuit: n-qubit CZ pairs, no SWAPs."""
    qc = QuantumCircuit(n)
    for i in range(0, n - 1, 2):
        qc.cz(i, i + 1)
    qc.h(range(n))
    return qc


# ---------------------------------------------------------------------------
# compile_superstaq — field correctness
# ---------------------------------------------------------------------------

@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_returns_compile_result(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert isinstance(r, CompileResult)


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_compiler_field_is_superstaq(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.compiler == "superstaq"


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_opt_level_is_minus_one(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.opt_level == -1


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_layout_method_is_auto(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.layout_method == "auto"


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_seed_is_minus_one(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.seed == -1


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_fidelity_is_minus_one(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.fidelity == -1.0


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_spec_fields_copied(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.spec_name == spec.name
    assert r.family == "qaoa"
    assert r.n_qubits == 4


# ---------------------------------------------------------------------------
# compile_superstaq — pre-compilation metrics come from original circuit
# ---------------------------------------------------------------------------

@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_two_q_before_from_original(mock_cls):
    # Ring n=4 → 4 edges × 1 RZZ = 4 two-qubit gates before compilation
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring", p=1)
    r = compile_superstaq(spec)

    assert r.two_q_before == 4


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_depth_before_positive(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.depth_before > 0


# ---------------------------------------------------------------------------
# compile_superstaq — post-compilation metrics from compiled circuit
# ---------------------------------------------------------------------------

@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_no_swaps_in_compiled(mock_cls):
    # Sqale has all-to-all connectivity; compiled output has no SWAP gates
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="er")
    r = compile_superstaq(spec)

    assert r.swap_count == 0


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_no_cx_in_compiled(mock_cls):
    # Sqale native gate is CZ, not CX
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="er")
    r = compile_superstaq(spec)

    assert r.cx_count == 0


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_compile_s_positive(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    r = compile_superstaq(spec)

    assert r.compile_s >= 0.0


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_routing_overhead_zero_when_no_2q_before(mock_cls):
    # BV n=2 with secret=0 has no oracle CX gates → two_q_before=0 → overhead=0.0
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(QuantumCircuit(2))

    spec = make_circuit("bv", n=2, secret=0)
    r = compile_superstaq(spec)

    assert r.two_q_before == 0
    assert r.routing_overhead == 0.0


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_uses_correct_target(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    compile_superstaq(spec, target="cq_sqale_qpu")

    mock_provider.cq_compile.assert_called_once()
    _, kwargs = mock_provider.cq_compile.call_args
    assert kwargs.get("target") == "cq_sqale_qpu"


# ---------------------------------------------------------------------------
# compile_superstaq — error handling
# ---------------------------------------------------------------------------

def test_compile_superstaq_raises_on_missing_key(monkeypatch):
    # Remove key from env; SuperstaqProvider raises EnvironmentError
    monkeypatch.delenv("SUPERSTAQ_API_KEY", raising=False)
    spec = make_circuit("qaoa", n=4, graph="ring")
    with pytest.raises(EnvironmentError):
        compile_superstaq(spec)


@patch("qiskit_superstaq.SuperstaqProvider")
def test_compile_superstaq_propagates_network_error(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.side_effect = RuntimeError("connection timeout")

    spec = make_circuit("qaoa", n=4, graph="ring")
    with pytest.raises(RuntimeError, match="connection timeout"):
        compile_superstaq(spec)


# ---------------------------------------------------------------------------
# run_superstaq_sweep — sweep behaviour
# ---------------------------------------------------------------------------

@patch("qiskit_superstaq.SuperstaqProvider")
def test_run_superstaq_sweep_returns_list(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("qaoa", n=4, graph="ring")
    results = run_superstaq_sweep([spec], verbose=False)

    assert isinstance(results, list)
    assert len(results) == 1


@patch("qiskit_superstaq.SuperstaqProvider")
def test_run_superstaq_sweep_one_result_per_spec(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    specs = [
        make_circuit("qaoa", n=4, graph="ring"),
        make_circuit("qft", n=4),
        make_circuit("bv", n=4),
    ]
    results = run_superstaq_sweep(specs, verbose=False)

    assert len(results) == 3


@patch("qiskit_superstaq.SuperstaqProvider")
def test_run_superstaq_sweep_continues_past_single_failure(mock_cls):
    # First spec fails, second succeeds → one result returned, not zero.
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.side_effect = [
        RuntimeError("API error"),
        _mock_output(_compiled_cz(4)),
    ]

    specs = [make_circuit("qaoa", n=4, graph="line"), make_circuit("qft", n=4)]
    results = run_superstaq_sweep(specs, verbose=False)

    assert len(results) == 1
    assert results[0].family == "qft"


@patch("qiskit_superstaq.SuperstaqProvider")
def test_run_superstaq_sweep_all_fail_returns_empty(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.side_effect = RuntimeError("API error")

    specs = [make_circuit("qaoa", n=4, graph="ring"), make_circuit("qft", n=4)]
    results = run_superstaq_sweep(specs, verbose=False)

    assert results == []


def test_run_superstaq_sweep_raises_on_missing_key(monkeypatch):
    # EnvironmentError propagates immediately rather than being swallowed.
    monkeypatch.delenv("SUPERSTAQ_API_KEY", raising=False)
    spec = make_circuit("qaoa", n=4, graph="ring")
    with pytest.raises(EnvironmentError):
        run_superstaq_sweep([spec], verbose=False)


@patch("qiskit_superstaq.SuperstaqProvider")
def test_run_superstaq_sweep_compiler_field_is_superstaq(mock_cls):
    mock_provider = MagicMock()
    mock_cls.return_value = mock_provider
    mock_provider.cq_compile.return_value = _mock_output(_compiled_cz(4))

    spec = make_circuit("clifford", n=4)
    results = run_superstaq_sweep([spec], verbose=False)

    assert results[0].compiler == "superstaq"


# ---------------------------------------------------------------------------
# Live integration tests (skipped unless RUN_SUPERSTAQ_LIVE=1)
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(
    not os.environ.get("RUN_SUPERSTAQ_LIVE"),
    reason="set RUN_SUPERSTAQ_LIVE=1 to run live SuperstaQ tests",
)


@live
def test_live_compile_superstaq_smoke():
    spec = make_circuit("qaoa", n=4, graph="er", p=1, seed=0)
    r = compile_superstaq(spec, target="cq_sqale_simulator")
    assert r.compiler == "superstaq"
    assert r.swap_count == 0
    assert r.two_q_after >= r.two_q_before
