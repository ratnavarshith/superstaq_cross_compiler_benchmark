"""Bernstein-Vazirani circuit generator parameterized by secret Hamming weight.

BV identifies a secret bitstring s in one quantum oracle query. The CX count
in the oracle equals the Hamming weight of s, so secret density is the real
routing difficulty knob. The existing quantum-compiler-bench implementation
hardcodes secret = all-ones, fixing CX count at n-1. This version exposes
the secret so we can sweep oracle sparsity and measure how compilers handle
different amounts of long-range entanglement.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def _resolve_secret(n: int, secret, seed: int) -> int:
    """Convert the `secret` argument to a concrete integer over n-1 bits."""
    input_bits = n - 1
    max_secret = (1 << input_bits) - 1

    if secret is None or secret == "all_ones":
        return max_secret

    if secret == "random":
        rng = np.random.default_rng(seed)
        # Integers in [1, max_secret] so the oracle always has at least one CX.
        return int(rng.integers(1, max_secret + 1))

    if isinstance(secret, int):
        if not (0 <= secret <= max_secret):
            raise ValueError(
                f"secret={secret} out of range for n={n} "
                f"(must be in [0, {max_secret}])."
            )
        return secret

    raise TypeError(
        f"secret must be None, 'all_ones', 'random', or int; "
        f"got {type(secret).__name__!r}."
    )


def make_bv(
    n: int,
    secret: "int | str | None" = None,
    seed: int = 0,
) -> QuantumCircuit:
    """Bernstein-Vazirani circuit on n qubits (n-1 input + 1 oracle ancilla).

    Includes terminal measurements. Strip them before fidelity computation
    (the benchmark harness handles this via CircuitSpec.has_measurements).

    Parameters
    ----------
    n : int
        Total qubit count. Input register: qubits 0..n-2. Ancilla: qubit n-1.
    secret : int | "all_ones" | "random" | None
        The secret bitstring as an integer over n-1 bits.
        None / "all_ones" → all bits set; CX count = n-1 (worst-case routing).
        int             → explicit secret; CX count = popcount(secret).
        "random"        → nonzero random integer (seeded), so oracle always
                          has at least one CX gate.
    seed : int
        Used only when secret="random".
    """
    s = _resolve_secret(n, secret, seed)
    input_bits = n - 1
    hw = bin(s).count("1")

    qc = QuantumCircuit(n, input_bits, name=f"bv_n{n}_hw{hw}")
    qc.x(n - 1)              # ancilla → |1>
    qc.h(range(n))           # input → |+>^(n-1), ancilla → |->
    for i in range(input_bits):
        if (s >> i) & 1:
            qc.cx(i, n - 1)  # oracle: flip ancilla for each set bit in s
    qc.h(range(input_bits))  # inverse Hadamard to read out s
    qc.measure(range(input_bits), range(input_bits))
    return qc
