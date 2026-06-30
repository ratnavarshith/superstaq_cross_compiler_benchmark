"""Backend selection with a safe fallback chain.

Mirrors quantum-compiler-bench/bench/backends.py with one change: the default
GenericBackendV2 size is 20 qubits rather than 16, giving headroom for circuits
up to n=12 with enough slack for routing to insert SWAPs.

Resolution order:
    1. FakeSherbrooke  (qiskit-ibm-runtime, 127-qubit heavy-hex, full noise)
    2. GenericBackendV2 (qiskit, configurable coupling map, synthetic noise)
    3. AerSimulator    (no coupling map, no noise model — last resort)

Every backend returned here is safe to pass to qiskit.transpile. Backends
that lack a noise model cause estimate_fidelity to return -1.0 rather than
crashing.
"""
from __future__ import annotations

AVAILABLE = ["fake_sherbrooke", "generic", "aer"]


def available_backends() -> list[str]:
    return list(AVAILABLE)


def _try_fake_sherbrooke():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke()
    except Exception:
        return None


def _try_generic(n_qubits: int = 20, seed: int = 42):
    """Linear coupling map of `n_qubits` qubits with a synthetic noise model."""
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except Exception:
        return None
    coupling = [[i, i + 1] for i in range(n_qubits - 1)]
    try:
        return GenericBackendV2(num_qubits=n_qubits, coupling_map=coupling, seed=seed)
    except TypeError:
        return GenericBackendV2(num_qubits=n_qubits, coupling_map=coupling)


def _aer():
    from qiskit_aer import AerSimulator
    return AerSimulator()


def load_backend(name: str = "auto", n_qubits: int = 20):
    """Return a backend usable as a transpile target.

    name:
        "auto"            try fake_sherbrooke → generic → aer
        "fake_sherbrooke" require qiskit-ibm-runtime; raise if missing
        "generic"         require GenericBackendV2; raise if missing
        "aer"             always succeeds; no coupling map, no noise model
    """
    name = name.lower()
    if name == "auto":
        return (_try_fake_sherbrooke() or _try_generic(n_qubits) or _aer())
    if name == "fake_sherbrooke":
        b = _try_fake_sherbrooke()
        if b is None:
            raise RuntimeError("FakeSherbrooke unavailable. Install qiskit-ibm-runtime.")
        return b
    if name == "generic":
        b = _try_generic(n_qubits)
        if b is None:
            raise RuntimeError("GenericBackendV2 unavailable. Update qiskit to >=1.0.")
        return b
    if name == "aer":
        return _aer()
    raise ValueError(
        f"Unknown backend: {name!r}. Choose from: auto, "
        "fake_sherbrooke, generic, aer."
    )


def backend_name(backend) -> str:
    """Best-effort human-readable name for a backend instance."""
    for attr in ("name", "backend_name"):
        v = getattr(backend, attr, None)
        if callable(v):
            try:
                return str(v())
            except Exception:
                pass
        elif v:
            return str(v)
    return type(backend).__name__
