"""QFT circuit generator with approximation-degree and decomposition control.

Two independent knobs matter for cross-compiler comparison:

  approximation_degree — drops the smallest controlled-phase rotations
      (angle < pi/2^k), reducing the O(n^2) two-qubit gate count toward
      O(n log n). This is a meaningful accuracy trade-off, not just
      parameter variation.

  decomposed — whether the compiler sees the QFT as a single high-level
      block or as elementary gates. Some compilers (SuperstaQ) may recognize
      and optimize the QFT block directly; others need the decomposed form.
      Passing both modes lets us test whether high-level recognition is
      actually exploited.

Implementation note: In Qiskit >=1.3, the recommended synthesis path for QFT
is qiskit.synthesis.qft.synth_qft_full, which returns a plain QuantumCircuit
of h/cp/swap gates and accepts approximation_degree. The old QFT BlueprintCircuit
is pending deprecation, and QFTGate only takes num_qubits (no approximation).
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full


def make_qft(
    n: int,
    approximation_degree: int = 0,
    decomposed: bool = True,
) -> QuantumCircuit:
    """Quantum Fourier Transform on n qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    approximation_degree : int
        Number of least-significant controlled-phase rotations to drop.
        0 → full QFT (O(n^2) 2Q gates).
        k → drops CP gates with angle < pi/2^(n-k), reducing gate count.
    decomposed : bool
        If True, return the circuit of elementary (h, cp, swap) gates so
        the compiler sees the individual structure. If False, wrap the QFT
        as a single composite gate — useful for compilers that recognize and
        optimize the QFT block directly.
    """
    gate_name = f"qft_n{n}_approx{approximation_degree}"

    if decomposed:
        return synth_qft_full(
            num_qubits=n,
            approximation_degree=approximation_degree,
            name=gate_name,
        )
    else:
        # Build the elementary circuit and wrap it as one composite gate so
        # the outer circuit has a single block instruction.
        inner = synth_qft_full(num_qubits=n, approximation_degree=approximation_degree)
        qc = QuantumCircuit(n, name=gate_name)
        qc.append(inner.to_gate(), range(n))
        return qc
