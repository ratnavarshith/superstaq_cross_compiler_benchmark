"""Tests for CircuitSpec and REGISTRY."""
import pytest
from scxb.spec import CircuitSpec, REGISTRY, available_families, make_circuit


def test_registry_has_four_families():
    assert set(REGISTRY.keys()) == {"qaoa", "qft", "bv", "clifford"}


def test_available_families_matches_registry():
    assert set(available_families()) == set(REGISTRY.keys())


@pytest.mark.parametrize("family", ["qaoa", "qft", "bv", "clifford"])
def test_make_circuit_returns_spec(family):
    spec = make_circuit(family, n=4)
    assert isinstance(spec, CircuitSpec)
    assert spec.n_qubits == 4
    assert spec.family == family


def test_make_circuit_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown circuit family"):
        make_circuit("hypercube", n=4)


def test_bv_spec_has_measurements():
    spec = make_circuit("bv", n=4)
    assert spec.has_measurements is True


@pytest.mark.parametrize("family", ["qaoa", "qft", "clifford"])
def test_non_bv_specs_have_no_measurements(family):
    spec = make_circuit(family, n=4)
    assert spec.has_measurements is False


def test_spec_params_carries_n_and_kwargs():
    spec = make_circuit("qaoa", n=5, graph="ring", p=2)
    assert spec.params["n"] == 5
    assert spec.params["graph"] == "ring"
    assert spec.params["p"] == 2


def test_spec_circuit_qubit_count_matches_n_qubits():
    for family in available_families():
        spec = make_circuit(family, n=4)
        assert spec.circuit.num_qubits == spec.n_qubits, (
            f"{family}: circuit.num_qubits={spec.circuit.num_qubits} "
            f"!= spec.n_qubits={spec.n_qubits}"
        )
