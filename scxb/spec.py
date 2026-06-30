"""CircuitSpec dataclass and REGISTRY of benchmark circuit families.

CircuitSpec bundles a circuit with its provenance — generator name, qubit
count, and the exact kwargs used to produce it. Every result row in a
cross-compiler benchmark can be traced back to the precise input, which
matters when a compiler produces a surprising output and you need to
reproduce the exact circuit.

The spec wrappers live here (not in each generator module) so that the
generator modules stay import-free of each other and circular imports don't
arise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qiskit import QuantumCircuit

from .bv import make_bv
from .clifford import make_clifford
from .qaoa import make_qaoa
from .qft import make_qft


@dataclass
class CircuitSpec:
    """A benchmark circuit bundled with its provenance."""
    name: str
    circuit: QuantumCircuit
    n_qubits: int
    family: str           # "qaoa" | "qft" | "bv" | "clifford"
    params: dict          # generator kwargs; stored for result-file traceability
    has_measurements: bool = False


def _qaoa_spec(n: int, **kwargs: Any) -> CircuitSpec:
    graph = kwargs.get("graph", "er")
    p = kwargs.get("p", 1)
    qc = make_qaoa(n, **kwargs)
    return CircuitSpec(
        name=f"qaoa_{graph}_n{n}_p{p}",
        circuit=qc,
        n_qubits=n,
        family="qaoa",
        params={"n": n, **kwargs},
        has_measurements=False,
    )


def _qft_spec(n: int, **kwargs: Any) -> CircuitSpec:
    approx = kwargs.get("approximation_degree", 0)
    qc = make_qft(n, **kwargs)
    return CircuitSpec(
        name=f"qft_n{n}_approx{approx}",
        circuit=qc,
        n_qubits=n,
        family="qft",
        params={"n": n, **kwargs},
        has_measurements=False,
    )


def _bv_spec(n: int, **kwargs: Any) -> CircuitSpec:
    qc = make_bv(n, **kwargs)
    return CircuitSpec(
        name=f"bv_n{n}",
        circuit=qc,
        n_qubits=n,
        family="bv",
        params={"n": n, **kwargs},
        has_measurements=True,
    )


def _clifford_spec(n: int, **kwargs: Any) -> CircuitSpec:
    qc = make_clifford(n, **kwargs)
    return CircuitSpec(
        name=f"clifford_n{n}",
        circuit=qc,
        n_qubits=n,
        family="clifford",
        params={"n": n, **kwargs},
        has_measurements=False,
    )


REGISTRY: dict[str, Callable[..., CircuitSpec]] = {
    "qaoa": _qaoa_spec,
    "qft": _qft_spec,
    "bv": _bv_spec,
    "clifford": _clifford_spec,
}


def make_circuit(family: str, n: int, **kwargs: Any) -> CircuitSpec:
    """Build a CircuitSpec by family name and qubit count."""
    if family not in REGISTRY:
        raise ValueError(
            f"Unknown circuit family: {family!r}. "
            f"Available: {', '.join(REGISTRY)}."
        )
    return REGISTRY[family](n, **kwargs)


def available_families() -> list[str]:
    return list(REGISTRY.keys())
