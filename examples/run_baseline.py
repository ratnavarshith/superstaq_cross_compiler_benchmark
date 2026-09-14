"""CLI: Qiskit transpiler baseline sweep for the cross-compiler benchmark.

Runs every circuit family through Qiskit's transpiler at varying optimization
levels and layout methods, records what comes out, and saves the results for
a future cross-compiler comparison against SuperstaQ.

Examples (PowerShell):

    # Default: all four families, n in {4, 6, 8}, opt 0-3, sabre layout
    python -m examples.run_baseline

    # Fast smoke test (small circuits, one layout, no fidelity)
    python -m examples.run_baseline --n-values 4 --opt-levels 0 3 `
        --layouts sabre --seeds 1 --no-fidelity

    # QAOA connectivity sweep only
    python -m examples.run_baseline --families qaoa --n-values 6 8 `
        --layouts sabre --opt-levels 3

    # All layout methods to compare SABRE vs trivial
    python -m examples.run_baseline --layouts trivial dense sabre `
        --n-values 4 6 --opt-levels 0 3 --no-fidelity

    # Generic backend (no qiskit-ibm-runtime needed)
    python -m examples.run_baseline --backend generic
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scxb.backends import available_backends, backend_name, load_backend
from scxb.plotting import generate_all
from scxb.sweep import make_default_specs, run_qiskit_sweep, save_results


_ALL_FAMILIES = ["qaoa", "qft", "bv", "clifford"]
_DEFAULT_QAOA_GRAPHS = ("line", "ring", "complete", "er")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Qiskit transpiler baseline for the SuperstaQ cross-compiler "
            "benchmark. Sweeps circuit families × opt levels × layout methods."
        ),
    )
    p.add_argument(
        "--backend", default="auto",
        choices=["auto"] + available_backends(),
        help="Compilation target. Default: auto (FakeSherbrooke → Generic → Aer).",
    )
    p.add_argument(
        "--families", nargs="+", default=None,
        choices=_ALL_FAMILIES,
        help="Circuit families to include. Default: all four.",
    )
    p.add_argument(
        "--graphs", nargs="+",
        default=list(_DEFAULT_QAOA_GRAPHS),
        help="QAOA graph connectivity types. Default: line ring complete er.",
    )
    p.add_argument(
        "--n-values", nargs="+", type=int, default=[4, 6, 8],
        help="Qubit counts to sweep.",
    )
    p.add_argument(
        "--opt-levels", nargs="+", type=int, default=[0, 1, 2, 3],
        help="Qiskit optimization_level values.",
    )
    p.add_argument(
        "--layouts", nargs="+", default=["sabre"],
        help="Layout methods. Default: sabre only (add trivial dense for full grid).",
    )
    p.add_argument(
        "--seeds", type=int, default=3,
        help="Transpiler seeds per cell; numeric metrics are the median.",
    )
    p.add_argument(
        "--fidelity-max-n", type=int, default=6,
        help="Skip fidelity for circuits wider than this (simulation is O(4^n)).",
    )
    p.add_argument(
        "--no-fidelity", action="store_true",
        help="Disable fidelity measurement entirely.",
    )
    p.add_argument(
        "--output-dir", default="results",
        help="Directory for results.json, results.csv, and plots.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (useful when matplotlib is not installed).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-cell progress output.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_qubits = max(args.n_values) + 4
    backend = load_backend(args.backend, n_qubits=n_qubits)
    print(f"backend : {backend_name(backend)}  "
          f"(n_qubits={getattr(backend, 'num_qubits', '?')})")
    print(f"n_values: {args.n_values}")
    print(f"families: {args.families or _ALL_FAMILIES}")
    print(f"layouts : {args.layouts}")
    print(f"opt     : {args.opt_levels}")
    print()

    specs = []
    for n in args.n_values:
        specs.extend(
            make_default_specs(
                n,
                families=args.families,
                qaoa_graphs=tuple(args.graphs),
            )
        )

    results = run_qiskit_sweep(
        specs,
        backend,
        opt_levels=tuple(args.opt_levels),
        layout_methods=tuple(args.layouts),
        n_seeds=args.seeds,
        measure_fidelity=not args.no_fidelity,
        fidelity_max_n=args.fidelity_max_n,
        verbose=not args.quiet,
    )

    paths = save_results(results, output_dir)
    print(f"\nsaved {paths['json']}")
    print(f"saved {paths['csv']}")

    if not args.no_plots:
        try:
            plot_paths = generate_all(results, output_dir)
            for p in plot_paths:
                print(f"saved {p}")
        except ImportError:
            print("(matplotlib/pandas not installed — plots skipped)")


if __name__ == "__main__":
    main()
