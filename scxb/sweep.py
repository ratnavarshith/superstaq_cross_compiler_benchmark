"""Sweep runner, default spec factory, and result persistence.

The sweep runner takes a flat list of CircuitSpecs and iterates over
(spec, opt_level, layout_method), taking the median over n_seeds for all
numeric metrics and recording fidelity from the first seed only.

`make_default_specs` is the canonical factory for the benchmark circuit set:
four QAOA graph families × n + QFT + BV + random Clifford. The CLI and tests
use this to ensure everyone is running the same set of circuits.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .compile import CompileResult, compile_qiskit, compile_superstaq
from .spec import CircuitSpec, make_circuit

# QAOA graph families in the default set. "random_regular" is excluded because
# n*degree must be even, which breaks for odd n and degree 3.
_DEFAULT_QAOA_GRAPHS = ("line", "ring", "complete", "er")

# QAOA graph families whose structure depends on a random seed (ER, random_regular).
# Deterministic families (line, ring, complete) are not in this set.
_RANDOM_QAOA_GRAPHS = frozenset({"er", "random_regular"})


def make_default_specs(
    n: int,
    families: "list[str] | None" = None,
    qaoa_graphs: "tuple[str, ...]" = _DEFAULT_QAOA_GRAPHS,
    qaoa_p: int = 1,
    seed: int = 0,
    n_random_seeds: int = 1,
) -> list[CircuitSpec]:
    """Build the standard benchmark spec list for qubit count `n`.

    This is the canonical circuit set for the cross-compiler benchmark. All
    callers (CLI, tests, analysis notebooks) should use this function to ensure
    they run identical inputs.

    Parameters
    ----------
    n : int
        Qubit count.
    families : list[str] | None
        Subset of families to include. None → all four.
    qaoa_graphs : tuple[str, ...]
        QAOA graph connectivity types.
    qaoa_p : int
        QAOA depth (number of cost+mixer layers).
    seed : int
        Seed for single-instance ER and Clifford (used when n_random_seeds=1).
    n_random_seeds : int
        Number of independent random instances to generate for ER QAOA and
        Clifford circuits. Seeds 0..n_random_seeds-1 are used; each instance
        gets a name suffix _s0, _s1, ... so rows are distinguishable in the
        results file. Deterministic families (line, ring, complete QAOA, QFT,
        BV) are always single-instance regardless of this value. Default 1
        preserves the original single-instance behaviour.
    """
    all_families = ["qaoa", "qft", "bv", "clifford"]
    active = set(families) if families is not None else set(all_families)
    specs: list[CircuitSpec] = []

    if "qaoa" in active:
        for graph in qaoa_graphs:
            if graph in _RANDOM_QAOA_GRAPHS and n_random_seeds > 1:
                for k in range(n_random_seeds):
                    spec = make_circuit("qaoa", n, graph=graph, p=qaoa_p, seed=k)
                    specs.append(replace(spec, name=f"{spec.name}_s{k}"))
            else:
                specs.append(make_circuit("qaoa", n, graph=graph, p=qaoa_p, seed=seed))

    if "qft" in active:
        specs.append(make_circuit("qft", n, approximation_degree=0, decomposed=True))

    if "bv" in active:
        specs.append(make_circuit("bv", n, secret="all_ones"))

    if "clifford" in active:
        if n_random_seeds > 1:
            for k in range(n_random_seeds):
                spec = make_circuit("clifford", n, seed=k)
                specs.append(replace(spec, name=f"{spec.name}_s{k}"))
        else:
            specs.append(make_circuit("clifford", n, seed=seed))

    return specs


def run_qiskit_sweep(
    specs: list[CircuitSpec],
    backend,
    opt_levels: "tuple[int, ...]" = (0, 1, 2, 3),
    layout_methods: "tuple[str, ...]" = ("trivial", "dense", "sabre"),
    n_seeds: int = 3,
    measure_fidelity: bool = True,
    fidelity_max_n: int = 6,
    verbose: bool = True,
) -> list[CompileResult]:
    """Sweep `(spec, opt_level, layout_method)` and return one result per cell.

    Each cell is the median over `n_seeds` transpiler runs; fidelity is taken
    from the first seed only (it is expensive and not seed-sensitive).
    Failed cells are skipped with a warning rather than crashing.
    """
    results: list[CompileResult] = []
    for spec in specs:
        for opt in opt_levels:
            for layout in layout_methods:
                seed_results: list[CompileResult] = []
                for s in range(n_seeds):
                    try:
                        r = compile_qiskit(
                            spec, backend,
                            opt_level=opt,
                            layout_method=layout,
                            seed=s,
                            measure_fidelity=(measure_fidelity and s == 0),
                            fidelity_max_n=fidelity_max_n,
                        )
                        seed_results.append(r)
                    except Exception as exc:
                        if verbose:
                            print(f"  skip {spec.name} opt={opt} "
                                  f"layout={layout} seed={s}: {exc}")
                        break

                if seed_results:
                    merged = _median_merge(seed_results)
                    results.append(merged)
                    if verbose:
                        print(
                            f"  {spec.name:30s}  opt={opt}  layout={layout:7s}  "
                            f"2Q: {merged.two_q_before}->{merged.two_q_after}  "
                            f"overhead={merged.routing_overhead:.2f}x  "
                            f"compile={merged.compile_s:.3f}s  "
                            f"fid={merged.fidelity:+.3f}"
                        )
    return results


def run_superstaq_sweep(
    specs: list[CircuitSpec],
    target: str = "cq_sqale_simulator",
    api_key: "str | None" = None,
    verbose: bool = True,
) -> list[CompileResult]:
    """Compile each spec through SuperstaQ and return one result per spec.

    No (opt_level × layout × seed) grid — SuperstaQ handles optimization and
    layout internally, and each call is a network round-trip. One result per spec.

    EnvironmentError (missing API key) propagates immediately; there is no point
    trying the next spec if auth fails. All other failures (network, rate limit,
    unsupported circuit) are caught per-spec so the sweep keeps going.

    Parameters
    ----------
    specs : list[CircuitSpec]
        Circuits to compile.
    target : str
        SuperstaQ backend string. Defaults to "cq_sqale_simulator".
    api_key : str | None
        SuperstaQ API key. None → reads SUPERSTAQ_API_KEY from the environment.
    verbose : bool
        Print one line per spec (or skip reason on failure).
    """
    results: list[CompileResult] = []
    for spec in specs:
        try:
            r = compile_superstaq(spec, target=target, api_key=api_key)
            results.append(r)
            if verbose:
                print(
                    f"  {spec.name:30s}  target={target}  "
                    f"2Q: {r.two_q_before}->{r.two_q_after}  "
                    f"overhead={r.routing_overhead:.2f}x  "
                    f"compile={r.compile_s:.3f}s"
                )
        except EnvironmentError:
            raise
        except Exception as exc:
            if verbose:
                print(f"  skip {spec.name}: {type(exc).__name__}: {exc}")
    return results


def _median_merge(rs: list[CompileResult]) -> CompileResult:
    """Median numeric metrics over seeds; fidelity from the first seed."""
    base = rs[0]

    def mi(field: str) -> int:
        return int(np.median([getattr(r, field) for r in rs]))

    def mf(field: str) -> float:
        return float(np.median([getattr(r, field) for r in rs]))

    return CompileResult(
        spec_name=base.spec_name,
        family=base.family,
        n_qubits=base.n_qubits,
        compiler=base.compiler,
        opt_level=base.opt_level,
        layout_method=base.layout_method,
        seed=-1,
        two_q_before=mi("two_q_before"),
        depth_before=mi("depth_before"),
        two_q_after=mi("two_q_after"),
        depth_after=mi("depth_after"),
        total_gates=mi("total_gates"),
        cx_count=mi("cx_count"),
        swap_count=mi("swap_count"),
        ecr_count=mi("ecr_count"),
        routing_overhead=mf("routing_overhead"),
        compile_s=mf("compile_s"),
        fidelity=base.fidelity,
    )


def save_results(
    results: list[CompileResult],
    output_dir: "str | Path",
) -> dict[str, Path]:
    """Persist results to `output_dir/results.json` and `results.csv`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    records = [asdict(r) for r in results]
    json_path.write_text(json.dumps(records, indent=2))

    if records:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return {"json": json_path, "csv": csv_path}


def load_results(path: "str | Path") -> list[CompileResult]:
    """Load results from a previously saved JSON or CSV file."""
    path = Path(path)
    if path.suffix == ".json":
        records = json.loads(path.read_text())
    elif path.suffix == ".csv":
        with path.open() as f:
            records = list(csv.DictReader(f))
        _int_fields = (
            "n_qubits", "opt_level", "seed",
            "two_q_before", "depth_before", "two_q_after", "depth_after",
            "total_gates", "cx_count", "swap_count", "ecr_count",
        )
        _float_fields = ("routing_overhead", "compile_s", "fidelity")
        for rec in records:
            for k in _int_fields:
                rec[k] = int(rec[k])
            for k in _float_fields:
                rec[k] = float(rec[k])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix!r}")
    return [CompileResult(**rec) for rec in records]
