"""QAOA circuit generators parameterized by graph connectivity family.

The key differentiator from quantum-compiler-bench: we sweep *connectivity
structure*, not just qubit count. This matters because SuperstaQ + neutral-atom
hardware can rearrange atoms on the fly (effectively all-to-all connectivity),
while a heavy-hex backend pays SWAP overhead for every non-adjacent pair.
Varying the graph family — from sparse lines to dense complete graphs — is how
we measure that gap quantitatively.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit


_GRAPH_FAMILIES = ("line", "ring", "complete", "er", "random_regular")


def _build_graph(
    n: int,
    graph: str,
    density: float,
    degree: int,
    seed: int,
) -> nx.Graph:
    if graph == "line":
        return nx.path_graph(n)
    if graph == "ring":
        return nx.cycle_graph(n)
    if graph == "complete":
        return nx.complete_graph(n)
    if graph == "er":
        return nx.erdos_renyi_graph(n, density, seed=seed)
    if graph == "random_regular":
        if (n * degree) % 2 != 0:
            raise ValueError(
                f"random_regular: n*degree must be even (got n={n}, degree={degree})."
            )
        return nx.random_regular_graph(degree, n, seed=seed)
    raise ValueError(
        f"Unknown graph family: {graph!r}. "
        f"Choose from: {', '.join(_GRAPH_FAMILIES)}."
    )


def graph_edge_count(
    n: int,
    graph: str,
    density: float = 0.5,
    degree: int = 3,
    seed: int = 0,
) -> int:
    """Exact edge count for a given graph family and parameters.

    Useful in tests to verify the circuit has the right number of RZZ gates
    without building the full circuit.
    """
    return _build_graph(n, graph, density, degree, seed).number_of_edges()


def make_qaoa(
    n: int,
    graph: str = "er",
    p: int = 1,
    density: float = 0.5,
    degree: int = 3,
    seed: int = 0,
) -> QuantumCircuit:
    """QAOA-style ansatz with a specific graph connectivity family.

    Uses fixed (non-optimized) RZZ + RX parameters seeded from `seed`. This
    is intentional: we are benchmarking the *compiler*, not the optimizer, so
    the angles just need to be realistic, not optimal.

    Parameters
    ----------
    n : int
        Number of qubits (= number of graph nodes).
    graph : str
        Connectivity family.
        "line"         — path graph, n-1 edges. Friendly to chain topologies.
        "ring"         — cycle graph, n edges. Moderate locality.
        "complete"     — K_n, n*(n-1)/2 edges. Worst case for fixed topologies.
        "er"           — Erdos-Renyi with `density`. Variable sparsity.
        "random_regular" — each node has exactly `degree` neighbors.
    p : int
        QAOA depth: number of cost+mixer layer pairs.
    density : float
        Edge probability for "er" graphs.
    degree : int
        Node degree for "random_regular" graphs.
    seed : int
        Controls both the graph structure (for "er" and "random_regular") and
        the random angle parameters. Same seed → identical circuit.
    """
    g = _build_graph(n, graph, density, degree, seed)
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, np.pi, p)
    betas = rng.uniform(0, np.pi / 2, p)

    qc = QuantumCircuit(n, name=f"qaoa_{graph}_n{n}_p{p}")
    qc.h(range(n))
    for k in range(p):
        for i, j in g.edges():
            qc.rzz(2 * gammas[k], i, j)
        for q in range(n):
            qc.rx(2 * betas[k], q)
    return qc
