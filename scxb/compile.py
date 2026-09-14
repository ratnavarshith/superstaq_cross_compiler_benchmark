"""Single-circuit compilation step for the cross-compiler benchmark.

`compile_qiskit` takes a CircuitSpec and returns a CompileResult that records
both pre- and post-compilation metrics. Keeping pre-compilation metrics in
the result (two_q_before, depth_before) matters for the cross-compiler story:
we need to compare what each compiler does to the *same* starting circuit.

`compile_superstaq` mirrors the same interface but targets Infleqtion's Sqale
neutral-atom hardware via the qiskit-superstaq SDK. Both functions populate
the same CompileResult dataclass, so analysis can do
df.groupby("compiler") directly.

Key difference in the numbers: Sqale's native gate is CZ (not CX/ECR) and its
all-to-all connectivity via atom shuttling means zero SWAP insertion. Expect
cx_count=swap_count=ecr_count=0 in SuperstaQ rows; the cross-compiler signal
lives in two_q_after (the gate-family-agnostic count) and swap_count=0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity

from .spec import CircuitSpec


@dataclass
class CompileResult:
    """One row of cross-compiler benchmark output.

    Carries circuit identity, compiler config, and pre+post metrics so every
    result is self-describing. `fidelity` is -1.0 when not measured (circuit
    has measurements, n too large, or backend has no noise model).
    """
    # Circuit identity
    spec_name: str
    family: str
    n_qubits: int
    # Compiler configuration
    compiler: str       # "qiskit" | "superstaq"
    opt_level: int      # Qiskit optimization_level (0–3); -1 for SuperstaQ default
    layout_method: str  # "trivial" | "dense" | "sabre" | "auto" | ""
    seed: int           # transpiler seed; -1 = median-aggregated across seeds
    # Pre-compilation metrics
    two_q_before: int
    depth_before: int
    # Post-compilation metrics
    two_q_after: int
    depth_after: int
    total_gates: int
    cx_count: int
    swap_count: int
    ecr_count: int
    routing_overhead: float  # two_q_after / two_q_before; 0.0 if no 2Q gates before
    compile_s: float
    fidelity: float          # state fidelity vs ideal; -1.0 if not measured


def _count_2q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def _ideal_sv(qc: QuantumCircuit) -> Statevector:
    stripped = qc.remove_final_measurements(inplace=False)
    return Statevector.from_instruction(stripped)


def _estimate_fidelity(original: QuantumCircuit, transpiled: QuantumCircuit, backend) -> float:
    """Fidelity between ideal statevector and noisy density-matrix simulation.

    Returns -1.0 if the backend has no noise model (AerSimulator bare), or
    if the simulation fails for any reason (circuit too wide, etc.).
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        return -1.0

    if isinstance(backend, AerSimulator):
        return -1.0

    try:
        sim = AerSimulator.from_backend(backend)
    except Exception:
        return -1.0

    try:
        ideal = _ideal_sv(original)
        stripped = transpiled.remove_final_measurements(inplace=False)
        stripped.save_density_matrix()
        result = sim.run(stripped, shots=1).result()
        rho = result.data(0)["density_matrix"]
        return float(state_fidelity(ideal, rho))
    except Exception:
        return -1.0


def compile_qiskit(
    spec: CircuitSpec,
    backend,
    opt_level: int = 3,
    layout_method: str = "sabre",
    seed: int = 0,
    measure_fidelity: bool = True,
    fidelity_max_n: int = 6,
) -> CompileResult:
    """Compile `spec.circuit` through Qiskit's transpiler and record metrics.

    Parameters
    ----------
    spec : CircuitSpec
        Circuit to compile. `spec.has_measurements` controls whether fidelity
        can be computed (BV circuits have measurements so fidelity is skipped).
    backend :
        Compilation target; typically from `scxb.backends.load_backend`.
    opt_level : int
        Qiskit optimization_level (0 = none, 3 = heaviest).
    layout_method : str
        Layout pass: "trivial", "dense", or "sabre".
    seed : int
        Transpiler seed (SABRE uses randomness; seeding makes runs comparable).
    measure_fidelity : bool
        Whether to run the density-matrix fidelity simulation.
    fidelity_max_n : int
        Skip fidelity for circuits wider than this (simulation is O(4^n)).
    """
    qc = spec.circuit

    t0 = time.perf_counter()
    tqc = transpile(
        qc,
        backend=backend,
        optimization_level=opt_level,
        layout_method=layout_method,
        seed_transpiler=seed,
    )
    compile_s = time.perf_counter() - t0

    two_q_before = _count_2q(qc)
    two_q_after = _count_2q(tqc)
    routing_overhead = (two_q_after / two_q_before) if two_q_before > 0 else 0.0

    ops = tqc.count_ops()

    fid = -1.0
    if (
        measure_fidelity
        and spec.n_qubits <= fidelity_max_n
        and not spec.has_measurements
    ):
        fid = _estimate_fidelity(qc, tqc, backend)

    return CompileResult(
        spec_name=spec.name,
        family=spec.family,
        n_qubits=spec.n_qubits,
        compiler="qiskit",
        opt_level=opt_level,
        layout_method=layout_method,
        seed=seed,
        two_q_before=two_q_before,
        depth_before=int(qc.depth()),
        two_q_after=two_q_after,
        depth_after=int(tqc.depth()),
        total_gates=int(sum(ops.values())),
        cx_count=int(ops.get("cx", 0)),
        swap_count=int(ops.get("swap", 0)),
        ecr_count=int(ops.get("ecr", 0)),
        routing_overhead=routing_overhead,
        compile_s=compile_s,
        fidelity=fid,
    )


def compile_superstaq(
    spec: CircuitSpec,
    target: str = "cq_sqale_simulator",
    api_key: "str | None" = None,
) -> CompileResult:
    """Compile `spec.circuit` through SuperstaQ targeting Sqale and record metrics.

    Uses provider.cq_compile() — the CQ-specific compilation path that maps to
    Sqale's native CZ gate set and all-to-all atom-shuttling connectivity.
    Target defaults to cq_sqale_simulator (no QPU credits consumed).

    Parameters
    ----------
    spec : CircuitSpec
        Circuit to compile. Passed as-is; SuperstaQ handles gate decomposition.
    target : str
        SuperstaQ backend string. "cq_sqale_simulator" (default) or "cq_sqale_qpu".
    api_key : str | None
        SuperstaQ API key. None → reads SUPERSTAQ_API_KEY from the environment.
        The key is never stored in the result or printed.

    Raises
    ------
    EnvironmentError
        If api_key is None and SUPERSTAQ_API_KEY is not set.
    Exception
        Network failures, rate limits, and unsupported circuits propagate so
        the caller (run_superstaq_sweep) can log and skip rather than crash.
    """
    try:
        import qiskit_superstaq as qss
    except ImportError as exc:
        raise ImportError(
            "qiskit-superstaq is required for compile_superstaq. "
            "Install it with: pip install qiskit-superstaq"
        ) from exc

    provider = qss.SuperstaqProvider(api_key=api_key)  # raises EnvironmentError if no key
    qc = spec.circuit

    t0 = time.perf_counter()
    output = provider.cq_compile(qc, target=target)
    compile_s = time.perf_counter() - t0

    tqc: QuantumCircuit = output.circuit  # single-circuit path → .circuit (not .circuits)

    two_q_before = _count_2q(qc)
    two_q_after = _count_2q(tqc)
    routing_overhead = (two_q_after / two_q_before) if two_q_before > 0 else 0.0

    ops = tqc.count_ops()

    return CompileResult(
        spec_name=spec.name,
        family=spec.family,
        n_qubits=spec.n_qubits,
        compiler="superstaq",
        opt_level=-1,
        layout_method="auto",
        seed=-1,
        two_q_before=two_q_before,
        depth_before=int(qc.depth()),
        two_q_after=two_q_after,
        depth_after=int(tqc.depth()),
        total_gates=int(sum(ops.values())),
        cx_count=int(ops.get("cx", 0)),
        swap_count=int(ops.get("swap", 0)),
        ecr_count=int(ops.get("ecr", 0)),
        routing_overhead=routing_overhead,
        compile_s=compile_s,
        fidelity=-1.0,
    )
