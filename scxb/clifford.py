"""Random Clifford circuit generator with optional depth control.

Two generation modes:

  depth=None  — sample a uniformly random Clifford group element via Qiskit's
      random_clifford(), convert to circuit. Dense, O(n^2) gates. Best for
      stress-testing the compiler on worst-case Clifford circuits.

  depth=d     — build a random circuit of exactly d Clifford generator gates
      (H, S, CX) on randomly chosen qubits. Shallower and more controllable.
      Useful for separating "compiler handles wide circuits" from "compiler
      handles deep circuits" without having the circuit size grow as O(n^2).

Both modes are deterministic given `seed`.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford


def make_clifford(
    n: int,
    depth: int | None = None,
    seed: int = 0,
) -> QuantumCircuit:
    """Random Clifford circuit on n qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    depth : int | None
        If None, sample a full random Clifford unitary (dense, O(n^2) gates
        after decomposition). If set, build a random circuit of exactly
        `depth` generator gates — controllable complexity.
    seed : int
        RNG seed. Same seed + same parameters → identical circuit.
    """
    if depth is None:
        qc = random_clifford(n, seed=seed).to_circuit()
        qc.name = f"clifford_n{n}"
        return qc

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"clifford_n{n}_d{depth}")
    for _ in range(depth):
        gate = int(rng.integers(3))  # 0=H, 1=S, 2=CX
        if gate == 0:
            q = int(rng.integers(n))
            qc.h(q)
        elif gate == 1:
            q = int(rng.integers(n))
            qc.s(q)
        else:  # CX — requires at least 2 qubits
            if n >= 2:
                ctrl = int(rng.integers(n))
                # Pick target != ctrl by sampling from {0..n-2} and shifting.
                tgt = int(rng.integers(n - 1))
                if tgt >= ctrl:
                    tgt += 1
                qc.cx(ctrl, tgt)
            else:
                # n=1 has no valid CX; fall back to H.
                qc.h(0)
    return qc
