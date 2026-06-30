"""CLI: Cross-compiler benchmark — SuperstaQ (Sqale) vs Qiskit (heavy-hex).

Runs the canonical circuit set through both compilers and saves all results to
a single JSON/CSV file where df.groupby("compiler") gives the side-by-side
comparison. The `compiler` column is "qiskit" or "superstaq" in every row.

Driving question: on the same circuits, does SuperstaQ's all-to-all neutral-atom
routing avoid the SWAP overhead that Qiskit inserts on heavy-hex connectivity?

Requires SUPERSTAQ_API_KEY in the environment for SuperstaQ compilation.
Use --no-superstaq to run Qiskit-only (e.g. while offline or for a quick check).

Examples (PowerShell):

    # Default: both compilers, all families, n in {4, 6, 8}, Qiskit opt=3 sabre
    python -m examples.run_cross_compiler

    # Fast smoke test — one n, one family, no plots
    python -m examples.run_cross_compiler --n-values 4 --families qaoa `
        --no-plots --quiet

    # Qiskit-only (no SuperstaQ key needed)
    python -m examples.run_cross_compiler --no-superstaq

    # Target real Sqale QPU instead of simulator (consumes credits)
    python -m examples.run_cross_compiler --target cq_sqale_qpu

    # Sweep Qiskit opt levels to see how SuperstaQ compares across the grid
    python -m examples.run_cross_compiler --opt-levels 0 1 2 3 `
        --layouts sabre --n-values 4 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scxb.backends import available_backends, backend_name, load_backend
from scxb.plotting import generate_all
from scxb.sweep import make_default_specs, run_qiskit_sweep, run_superstaq_sweep, save_results


_ALL_FAMILIES = ["qaoa", "qft", "bv", "clifford"]
_DEFAULT_QAOA_GRAPHS = ("line", "ring", "complete", "er")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-compiler benchmark: SuperstaQ (Sqale neutral-atom) vs "
            "Qiskit (heavy-hex). Saves combined results for groupby('compiler') analysis."
        ),
    )
    # --- Qiskit side ---
    p.add_argument(
        "--backend", default="auto",
        choices=["auto"] + available_backends(),
        help="Qiskit compilation target. Default: auto (FakeSherbrooke → Generic → Aer).",
    )
    p.add_argument(
        "--opt-levels", nargs="+", type=int, default=[3],
        help="Qiskit optimization_level values. Default: 3 (best) for a fair comparison.",
    )
    p.add_argument(
        "--layouts", nargs="+", default=["sabre"],
        help="Qiskit layout methods. Default: sabre.",
    )
    p.add_argument(
        "--seeds", type=int, default=3,
        help="Qiskit transpiler seeds per cell; numeric metrics are the median.",
    )
    p.add_argument(
        "--no-fidelity", action="store_true",
        help="Disable Qiskit fidelity measurement (SuperstaQ fidelity is always -1.0).",
    )
    p.add_argument(
        "--fidelity-max-n", type=int, default=6,
        help="Skip fidelity for circuits wider than this (simulation is O(4^n)).",
    )
    # --- SuperstaQ side ---
    p.add_argument(
        "--target", default="cq_sqale_simulator",
        help=(
            "SuperstaQ backend target. Default: cq_sqale_simulator "
            "(no QPU credits). Use cq_sqale_qpu for real hardware."
        ),
    )
    p.add_argument(
        "--no-superstaq", action="store_true",
        help="Skip SuperstaQ compilation (Qiskit-only run, no API key needed).",
    )
    # --- Circuit set ---
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
        "--n-random-seeds", type=int, default=5,
        help=(
            "Independent random instances for ER QAOA and Clifford. "
            "Each instance is compiled separately; median is taken at analysis time. "
            "Default 5. Set to 1 for the original single-instance behaviour."
        ),
    )
    # --- Output ---
    p.add_argument(
        "--output-dir", default="results",
        help="Directory for results.json, results.csv, and plots.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-circuit progress output.",
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
    print(f"target  : {args.target}")
    print(f"n_values: {args.n_values}")
    print(f"families: {args.families or _ALL_FAMILIES}")
    print(f"opt     : {args.opt_levels}  layouts: {args.layouts}")
    print(f"random seeds (ER/Clifford): {args.n_random_seeds}")
    print(f"superstaq: {'disabled (--no-superstaq)' if args.no_superstaq else 'enabled'}")
    print()

    specs = []
    for n in args.n_values:
        specs.extend(
            make_default_specs(
                n,
                families=args.families,
                qaoa_graphs=tuple(args.graphs),
                n_random_seeds=args.n_random_seeds,
            )
        )

    all_results = []

    # --- Qiskit sweep ---
    print("=== Qiskit transpiler ===")
    qiskit_results = run_qiskit_sweep(
        specs,
        backend,
        opt_levels=tuple(args.opt_levels),
        layout_methods=tuple(args.layouts),
        n_seeds=args.seeds,
        measure_fidelity=not args.no_fidelity,
        fidelity_max_n=args.fidelity_max_n,
        verbose=not args.quiet,
    )
    all_results.extend(qiskit_results)
    print(f"\nQiskit: {len(qiskit_results)} results\n")

    # --- SuperstaQ sweep ---
    if not args.no_superstaq:
        print(f"=== SuperstaQ ({args.target}) ===")
        try:
            superstaq_results = run_superstaq_sweep(
                specs,
                target=args.target,
                verbose=not args.quiet,
            )
            all_results.extend(superstaq_results)
            print(f"\nSuperstaQ: {len(superstaq_results)} results\n")
        except EnvironmentError as exc:
            print(f"\n[warning] SuperstaQ skipped - {exc}", file=sys.stderr)
            print("[warning] Set SUPERSTAQ_API_KEY or use --no-superstaq.\n",
                  file=sys.stderr)

    if not all_results:
        print("No results collected — nothing saved.", file=sys.stderr)
        sys.exit(1)

    # --- Save combined ---
    paths = save_results(all_results, output_dir)
    print(f"saved {paths['json']}  ({len(all_results)} rows)")
    print(f"saved {paths['csv']}")

    compilers = sorted({r.compiler for r in all_results})
    print(f"compilers in output: {compilers}")

    # --- Plots ---
    if not args.no_plots:
        try:
            plot_paths = generate_all(all_results, output_dir)
            for p in plot_paths:
                print(f"saved {p}")
        except ImportError:
            print("(matplotlib/pandas not installed — plots skipped)")


if __name__ == "__main__":
    main()
