"""Plots for the cross-compiler baseline results.

Three baseline plots work on Qiskit-only data:

  plot_routing_overhead_by_connectivity — routing overhead vs QAOA graph
      connectivity family (line < ring < ER < complete), making the
      heavy-hex SWAP cost visible.

  plot_two_q_inflation — 2Q gates before vs after compilation for each
      circuit family, showing how much the transpiler inflates each type.

  plot_compile_time — wall time vs qubit count per family+layout combination.

Three more (plot_qaoa_overhead_vs_connectivity, plot_overhead_by_family,
plot_compile_time_crosscompiler) kick in once SuperstaQ rows are present and
overlay both compilers directly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sweep import load_results
from .compile import CompileResult

_GRAPH_ORDER = ("line", "ring", "er", "complete", "random_regular")

# Colors and labels shared by the cross-compiler plots
_C_COLOR = {"qiskit": "#1f77b4", "superstaq": "#ff7f0e"}
_C_LABEL = {
    "qiskit":    "Qiskit  (heavy-hex / ECR)",
    "superstaq": "SuperstaQ  (Sqale / CZ)",
}
_C_MARKER = {"qiskit": "o", "superstaq": "s"}
_C_LINE   = {"qiskit": "-", "superstaq": "--"}


def _to_df(results_or_path) -> pd.DataFrame:
    if isinstance(results_or_path, (str, Path)):
        results = load_results(results_or_path)
    else:
        results = results_or_path
    return pd.DataFrame([r.__dict__ for r in results])


def _agg_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (group_key, n_qubits, family, compiler) with median + IQR.

    For ER QAOA and Clifford, which may have multiple _s{k} instances, this
    collapses all seeds to one group row. Deterministic families have n_instances=1,
    so IQR collapses to zero (q25 == q75 == median).

    group_key: QAOA graph name ("er", "ring", …) for QAOA; family name otherwise.
    """
    df = df.copy()
    df["group_key"] = df.apply(
        lambda r: r["spec_name"].split("_")[1] if r["family"] == "qaoa" else r["family"],
        axis=1,
    )
    return (
        df.groupby(["group_key", "family", "n_qubits", "compiler"])
        .agg(
            overhead_median=("routing_overhead", "median"),
            overhead_q25=("routing_overhead", lambda x: float(np.percentile(x, 25))),
            overhead_q75=("routing_overhead", lambda x: float(np.percentile(x, 75))),
            two_q_before=("two_q_before", "median"),
            compile_s=("compile_s", "median"),
            n_instances=("routing_overhead", "count"),
        )
        .reset_index()
    )


def plot_routing_overhead_by_connectivity(
    results_or_path,
    output_dir: "str | Path",
    layout_method: str = "sabre",
    opt_level: int = 3,
) -> "Path | None":
    """Routing overhead vs QAOA graph family at the best compiler settings.

    This is the headline plot: it shows how much SWAP overhead the Qiskit
    transpiler imposes for each connectivity class. The ordering
    line < ring < ER < complete follows from edge density. Neutral-atom
    hardware can rearrange atoms, so this overhead is hardware-dependent; see
    plot_qaoa_overhead_vs_connectivity for the side-by-side with SuperstaQ.
    """
    df = _to_df(results_or_path)
    if df.empty:
        return None

    qaoa = df[
        (df["family"] == "qaoa")
        & (df["layout_method"] == layout_method)
        & (df["opt_level"] == opt_level)
    ].copy()
    if qaoa.empty:
        return None

    # Extract graph family from spec_name: "qaoa_{graph}_n{n}_p{p}"
    qaoa["graph"] = qaoa["spec_name"].str.split("_").str[1]
    graph_order = [g for g in _GRAPH_ORDER if g in qaoa["graph"].unique()]
    n_values = sorted(qaoa["n_qubits"].unique())

    # Aggregate by (graph, n_qubits) so multiple ER seeds collapse to one median.
    qaoa_agg = (
        qaoa.groupby(["graph", "n_qubits"])["routing_overhead"]
        .median()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for n in n_values:
        sub = qaoa_agg[qaoa_agg["n_qubits"] == n]
        sub = sub.set_index("graph").reindex(graph_order)
        overhead = sub["routing_overhead"].values
        ax.plot(range(len(graph_order)), overhead, "o-", label=f"n={n}", alpha=0.85)

    ax.set_xticks(range(len(graph_order)))
    ax.set_xticklabels(graph_order)
    ax.set_xlabel("QAOA graph connectivity")
    ax.set_ylabel("routing overhead (compiled 2Q / original 2Q)")
    ax.set_title(
        f"SWAP overhead grows with connectivity  "
        f"(layout={layout_method}, opt={opt_level})"
    )
    ax.legend(title="qubits", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="no overhead")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir) / "routing_overhead_by_connectivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_two_q_inflation(
    results_or_path,
    output_dir: "str | Path",
    layout_method: str = "sabre",
) -> "Path | None":
    """2Q gates before vs after compilation, by circuit family and opt level.

    Shows the absolute 2Q gate inflation per family so we can see which
    circuit types are hurt most by routing on a constrained topology.
    """
    df = _to_df(results_or_path)
    if df.empty:
        return None

    sub = df[df["layout_method"] == layout_method]
    if sub.empty:
        return None

    families = sorted(sub["family"].unique())
    n_values = sorted(sub["n_qubits"].unique())

    fig, axes = plt.subplots(1, len(families),
                             figsize=(5 * len(families), 4),
                             sharey=False)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        seg = sub[sub["family"] == family]
        for n in n_values:
            subseg = seg[seg["n_qubits"] == n].sort_values("opt_level")
            if subseg.empty:
                continue
            ax.plot(
                subseg["opt_level"], subseg["two_q_after"],
                "o-", label=f"n={n}", alpha=0.85,
            )
            # Mark the before-compilation level as a dashed horizontal line
            before = subseg["two_q_before"].iloc[0]
            ax.axhline(before, linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
        ax.set_title(family)
        ax.set_xlabel("opt level")
        ax.legend(title="qubits", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("2Q gates after compilation")
    fig.suptitle(f"2Q gate inflation (layout={layout_method}; dashed = before)")
    fig.tight_layout()

    out = Path(output_dir) / "two_q_inflation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_compile_time(
    results_or_path,
    output_dir: "str | Path",
    opt_level: int = 3,
    layout_method: str = "sabre",
) -> "Path | None":
    """Compile time vs n_qubits, by circuit family."""
    df = _to_df(results_or_path)
    if df.empty:
        return None

    sub = df[
        (df["opt_level"] == opt_level)
        & (df["layout_method"] == layout_method)
        & (df["seed"] == -1)  # aggregated rows
    ]
    if sub.empty:
        sub = df[
            (df["opt_level"] == opt_level)
            & (df["layout_method"] == layout_method)
        ]
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for family, s in sub.groupby("family"):
        s = s.sort_values("n_qubits")
        ax.plot(s["n_qubits"], s["compile_s"], "o-", label=family, alpha=0.85)

    ax.set_xlabel("qubits (n)")
    ax.set_ylabel("compile time (s, median over seeds)")
    ax.set_title(f"Transpiler runtime (opt={opt_level}, layout={layout_method})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir) / "compile_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_qaoa_overhead_vs_connectivity(
    results_or_path,
    output_dir: "str | Path",
) -> "Path | None":
    """Headline plot: QAOA routing overhead by graph connectivity, both compilers.

    Shows one sub-plot per n, with Qiskit (solid) and SuperstaQ (dashed) overhead
    on the same axes. The ring case is where topology-conditional advantage appears:
    heavy-hex cannot route ring-structured gates without extra ECR gates, while
    Sqale's all-to-all atom shuttling keeps overhead at the CZ-decomposition floor.

    Returns None when fewer than two compilers are present (Qiskit-only run).
    """
    df = _to_df(results_or_path)
    if df.empty or df["compiler"].nunique() < 2:
        return None

    qaoa = df[df["family"] == "qaoa"].copy()
    if qaoa.empty:
        return None

    qaoa["graph"] = qaoa["spec_name"].str.split("_").str[1]
    graph_order = [g for g in _GRAPH_ORDER if g in qaoa["graph"].unique()]
    n_values = sorted(qaoa["n_qubits"].unique())

    # Aggregate: one median + IQR per (graph, n_qubits, compiler).
    # Single-instance graphs (line, ring, complete) have q25==q75==median so their
    # IQR band is invisible. Multi-seed graphs (er) show a shaded IQR band.
    agg = (
        qaoa.groupby(["graph", "n_qubits", "compiler"])
        .agg(
            median=("routing_overhead", "median"),
            q25=("routing_overhead", lambda x: float(np.percentile(x, 25))),
            q75=("routing_overhead", lambda x: float(np.percentile(x, 75))),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(
        1, len(n_values),
        figsize=(4.5 * len(n_values), 4),
        sharey=True,
    )
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        for compiler in ["qiskit", "superstaq"]:
            sub = agg[(agg["n_qubits"] == n) & (agg["compiler"] == compiler)]
            if sub.empty:
                continue
            sub = sub.set_index("graph").reindex(graph_order)
            xs = list(range(len(graph_order)))
            medians = sub["median"].values
            q25 = sub["q25"].fillna(sub["median"]).values
            q75 = sub["q75"].fillna(sub["median"]).values
            color = _C_COLOR[compiler]
            ax.plot(
                xs, medians,
                marker=_C_MARKER[compiler],
                linestyle=_C_LINE[compiler],
                color=color,
                label=_C_LABEL[compiler],
                alpha=0.9,
                linewidth=1.8,
            )
            # IQR shading: invisible for single-instance (q25==q75), subtle for er
            ax.fill_between(xs, q25, q75, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"n={n} qubits")
        ax.set_xticks(range(len(graph_order)))
        ax.set_xticklabels(graph_order)
        ax.set_xlabel("QAOA graph connectivity")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("routing overhead  (compiled 2Q / original 2Q)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        "QAOA routing overhead by connectivity: Qiskit (heavy-hex) vs SuperstaQ (Sqale)\n"
        "SuperstaQ advantage appears on ring topology; compilers converge on dense graphs",
        fontsize=10,
    )
    fig.tight_layout()

    out = Path(output_dir) / "qaoa_overhead_vs_connectivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_overhead_by_family(
    results_or_path,
    output_dir: "str | Path",
    n: int = 8,
) -> "Path | None":
    """Grouped bar chart: routing overhead for all circuit types at n qubits.

    Shows where each compiler wins. The split is visible at a glance:
    SuperstaQ wins on ring-topology QAOA; Qiskit opt-3 wins on QFT and Clifford.
    QAOA circuits are ordered line < ring < er < complete by edge density.
    """
    df = _to_df(results_or_path)
    if df.empty or df["compiler"].nunique() < 2:
        return None

    sub_df = df[df["n_qubits"] == n]
    if sub_df.empty:
        n = int(df["n_qubits"].max())
        sub_df = df[df["n_qubits"] == n]
    if sub_df.empty:
        return None

    # Aggregate: one median+IQR per (group_key, n_qubits, compiler).
    # ER QAOA and Clifford collapse from N seed rows to 1 group row.
    agg = _agg_by_group(sub_df)
    agg_n = agg[agg["n_qubits"] == n]

    # Canonical group order: QAOA by edge density, then QFT, BV, Clifford
    _GROUP_ORDER = [
        ("line", "qaoa"), ("ring", "qaoa"), ("er", "qaoa"), ("complete", "qaoa"),
        ("qft", "qft"), ("bv", "bv"), ("clifford", "clifford"),
    ]
    groups_present = list({
        (row.group_key, row.family)
        for row in agg_n.itertuples()
    })
    groups = [g for g in _GROUP_ORDER if g in groups_present]

    def _short_group(group_key: str, family: str) -> str:
        if family == "qaoa":
            return f"QAOA\n({group_key})"
        return {"qft": "QFT", "bv": "BV", "clifford": "Clifford"}.get(group_key, group_key)

    labels = [_short_group(gk, fam) for gk, fam in groups]
    x = range(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 1.1), 5))

    for compiler, offset in [("qiskit", -width / 2), ("superstaq", width / 2)]:
        comp = agg_n[agg_n["compiler"] == compiler]
        heights, q25_arr, q75_arr = [], [], []
        for gk, fam in groups:
            row = comp[(comp["group_key"] == gk) & (comp["family"] == fam)]
            if row.empty:
                heights.append(0.0); q25_arr.append(0.0); q75_arr.append(0.0)
            else:
                h = float(row["overhead_median"].iloc[0])
                heights.append(h)
                q25_arr.append(float(row["overhead_q25"].iloc[0]))
                q75_arr.append(float(row["overhead_q75"].iloc[0]))
        heights = np.array(heights)
        yerr = np.array([heights - np.array(q25_arr), np.array(q75_arr) - heights])
        ax.bar(
            [xi + offset for xi in x],
            heights,
            width,
            label=_C_LABEL[compiler],
            color=_C_COLOR[compiler],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 1.2, "alpha": 0.7},
        )

    # Separator between QAOA group and other families
    n_qaoa = sum(1 for _, fam in groups if fam == "qaoa")
    if 0 < n_qaoa < len(groups):
        ax.axvline(n_qaoa - 0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("routing overhead  (compiled 2Q / original 2Q)")
    ax.set_title(
        f"Overhead by circuit type (n={n}): where each compiler wins\n"
        "SuperstaQ wins ring QAOA; Qiskit opt-3 wins QFT and Clifford",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir) / "overhead_by_family.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_compile_time_crosscompiler(
    results_or_path,
    output_dir: "str | Path",
) -> "Path | None":
    """Compile time vs qubit count: Qiskit (~25 ms) vs SuperstaQ (~1 s, network round-trip).

    Log-scale y-axis makes the 20-40x difference visible. The gap is not a
    compiler quality issue — it is the fixed cost of an HTTPS API call per circuit.
    """
    df = _to_df(results_or_path)
    if df.empty or df["compiler"].nunique() < 2:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    n_values = sorted(df["n_qubits"].unique())

    for compiler in ["qiskit", "superstaq"]:
        comp = df[df["compiler"] == compiler]
        if comp.empty:
            continue
        # Median compile_s per n across all families/specs at that n
        by_n = comp.groupby("n_qubits")["compile_s"].median().reset_index()
        by_n = by_n.sort_values("n_qubits")
        ax.plot(
            by_n["n_qubits"], by_n["compile_s"],
            marker=_C_MARKER[compiler],
            linestyle=_C_LINE[compiler],
            color=_C_COLOR[compiler],
            label=_C_LABEL[compiler],
            linewidth=2,
            markersize=7,
        )
        # Individual family lines as faint background traces
        for _, fam_df in comp.groupby("family"):
            fam_df = fam_df.groupby("n_qubits")["compile_s"].median().reset_index()
            fam_df = fam_df.sort_values("n_qubits")
            ax.plot(
                fam_df["n_qubits"], fam_df["compile_s"],
                color=_C_COLOR[compiler], alpha=0.15, linewidth=0.8,
            )

    ax.set_yscale("log")
    ax.set_xlabel("qubits (n)")
    ax.set_ylabel("compile time (s, log scale)")
    ax.set_title(
        "Compile time: Qiskit (local) vs SuperstaQ (HTTPS API round-trip)\n"
        "Gap is ~20-40x — network latency, not compiler quality",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(n_values)
    fig.tight_layout()

    out = Path(output_dir) / "compile_time_crosscompiler.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_summary_table(
    results_or_path,
    output_dir: "str | Path",
) -> "pd.DataFrame | None":
    """Build and save a winner-per-circuit summary table.

    Columns: spec_name, n_qubits, family, two_q_before,
             qiskit_overhead, superstaq_overhead, winner, advantage_pct,
             compile_speedup (how many times faster Qiskit is to compile).

    Winner is "tie" when the overhead difference is within 5 % of the larger.
    Saved to output_dir/summary_table.csv and returned as a DataFrame.
    """
    df = _to_df(results_or_path)
    if df.empty or df["compiler"].nunique() < 2:
        return None

    # Aggregate seeds first so ER/Clifford collapse to one row per (group, n, compiler)
    agg = _agg_by_group(df)

    qk = (
        agg[agg["compiler"] == "qiskit"]
        [["group_key", "family", "n_qubits", "two_q_before",
          "overhead_median", "compile_s", "n_instances"]]
        .rename(columns={
            "overhead_median": "qiskit_overhead",
            "compile_s": "qiskit_compile_s",
            "n_instances": "n_seeds",
        })
    )
    sq = (
        agg[agg["compiler"] == "superstaq"]
        [["group_key", "n_qubits", "overhead_median", "compile_s"]]
        .rename(columns={
            "overhead_median": "superstaq_overhead",
            "compile_s": "superstaq_compile_s",
        })
    )
    tbl = qk.merge(sq, on=["group_key", "n_qubits"]).sort_values(
        ["family", "n_qubits", "group_key"]
    )

    def _winner(row) -> str:
        q, s = row["qiskit_overhead"], row["superstaq_overhead"]
        denom = max(q, s, 1e-9)
        if abs(q - s) / denom < 0.05:
            return "tie"
        return "qiskit" if q < s else "superstaq"

    tbl["winner"] = tbl.apply(_winner, axis=1)
    tbl["advantage_pct"] = (
        (tbl["qiskit_overhead"] - tbl["superstaq_overhead"]).abs()
        / tbl[["qiskit_overhead", "superstaq_overhead"]].max(axis=1)
        * 100
    ).round(1)
    tbl["compile_speedup"] = (
        tbl["superstaq_compile_s"] / tbl["qiskit_compile_s"]
    ).round(1)

    tbl = tbl.drop(columns=["qiskit_compile_s", "superstaq_compile_s"])
    tbl = tbl.round({"qiskit_overhead": 3, "superstaq_overhead": 3})

    out = Path(output_dir) / "summary_table.csv"
    tbl.to_csv(out, index=False)
    return tbl


def generate_all(
    results_or_path,
    output_dir: "str | Path",
) -> list[Path]:
    """Run the baseline plots, plus cross-compiler plots when both compilers are present."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Baseline plots (work on single-compiler data too)
    for fn in (
        plot_routing_overhead_by_connectivity,
        plot_two_q_inflation,
        plot_compile_time,
    ):
        p = fn(results_or_path, output_dir)
        if p is not None:
            paths.append(p)

    # Cross-compiler plots — only when both compilers are in the data
    df = _to_df(results_or_path)
    if df["compiler"].nunique() >= 2:
        for fn in (
            plot_qaoa_overhead_vs_connectivity,
            plot_overhead_by_family,
            plot_compile_time_crosscompiler,
        ):
            p = fn(results_or_path, output_dir)
            if p is not None:
                paths.append(p)
        tbl = make_summary_table(results_or_path, output_dir)
        if tbl is not None:
            paths.append(output_dir / "summary_table.csv")

    return paths
