"""Tests for make_default_specs, run_qiskit_sweep, save_results, load_results.

Uses AerSimulator so tests are fast and environment-independent. Because Aer
has no coupling map, SWAP insertion doesn't happen and routing_overhead is
deterministic (1.0 for circuits with 2Q gates, 0.0 for circuits without).
Tests focus on:
  - make_default_specs returns the right circuits and families
  - run_qiskit_sweep produces one result per (spec, opt, layout) cell
  - save/load roundtrip preserves all fields exactly
"""
import json
import pytest
from pathlib import Path
from qiskit_aer import AerSimulator

from scxb.compile import CompileResult
from scxb.spec import make_circuit
from scxb.sweep import (
    load_results,
    make_default_specs,
    run_qiskit_sweep,
    save_results,
)


@pytest.fixture(scope="module")
def aer():
    return AerSimulator()


# --- make_default_specs ---

def test_make_default_specs_returns_nonempty_list():
    specs = make_default_specs(4)
    assert len(specs) > 0


def test_make_default_specs_covers_four_families():
    specs = make_default_specs(4)
    families = {s.family for s in specs}
    assert families == {"qaoa", "qft", "bv", "clifford"}


def test_make_default_specs_qaoa_graphs_count():
    specs = make_default_specs(4, families=["qaoa"])
    # default _DEFAULT_QAOA_GRAPHS has 4 entries (line, ring, complete, er)
    assert len(specs) == 4


def test_make_default_specs_all_n_qubits_correct():
    for n in (4, 6):
        for spec in make_default_specs(n):
            assert spec.n_qubits == n, (
                f"{spec.name}: expected n_qubits={n}, got {spec.n_qubits}"
            )


def test_make_default_specs_family_filter():
    specs = make_default_specs(4, families=["bv", "clifford"])
    families = {s.family for s in specs}
    assert families == {"bv", "clifford"}


def test_make_default_specs_n_random_seeds_1_unchanged():
    # Default n_random_seeds=1 produces the same count as before the parameter existed.
    specs = make_default_specs(4)
    assert len(specs) == 7  # 4 qaoa + 1 qft + 1 bv + 1 clifford


def test_make_default_specs_n_random_seeds_er_count():
    specs = make_default_specs(4, families=["qaoa"], n_random_seeds=3)
    # line + ring + complete (deterministic) + 3 ER = 6
    assert len(specs) == 6


def test_make_default_specs_n_random_seeds_clifford_count():
    specs = make_default_specs(4, families=["clifford"], n_random_seeds=3)
    assert len(specs) == 3


def test_make_default_specs_n_random_seeds_er_name_suffix():
    specs = make_default_specs(4, families=["qaoa"], n_random_seeds=3)
    er_names = sorted(s.name for s in specs if "_er_" in s.name)
    assert er_names == ["qaoa_er_n4_p1_s0", "qaoa_er_n4_p1_s1", "qaoa_er_n4_p1_s2"]


def test_make_default_specs_n_random_seeds_clifford_name_suffix():
    specs = make_default_specs(4, families=["clifford"], n_random_seeds=3)
    names = sorted(s.name for s in specs)
    assert names == ["clifford_n4_s0", "clifford_n4_s1", "clifford_n4_s2"]


def test_make_default_specs_n_random_seeds_deterministic_no_suffix():
    # Deterministic families (line, ring, complete, qft, bv) never get _s suffix.
    specs = make_default_specs(4, n_random_seeds=3)
    det_names = [s.name for s in specs
                 if s.family in ("qft", "bv")
                 or (s.family == "qaoa" and "_line_" in s.name)
                 or (s.family == "qaoa" and "_ring_" in s.name)
                 or (s.family == "qaoa" and "_complete_" in s.name)]
    assert all("_s" not in n for n in det_names)


def test_make_default_specs_n_random_seeds_total_count():
    # With n_random_seeds=5: 3 det-qaoa + 5 er + 1 qft + 1 bv + 5 clifford = 15
    specs = make_default_specs(4, n_random_seeds=5)
    assert len(specs) == 15


# --- run_qiskit_sweep ---

def test_run_qiskit_sweep_returns_list(aer):
    spec = make_circuit("qaoa", n=4, graph="ring")
    results = run_qiskit_sweep(
        [spec], aer,
        opt_levels=(0,),
        layout_methods=("sabre",),
        n_seeds=1,
        measure_fidelity=False,
        verbose=False,
    )
    assert isinstance(results, list)
    assert len(results) > 0


def test_run_qiskit_sweep_result_count(aer):
    # 1 spec × 2 opt_levels × 2 layouts = 4 results
    spec = make_circuit("qaoa", n=4, graph="ring")
    results = run_qiskit_sweep(
        [spec], aer,
        opt_levels=(0, 3),
        layout_methods=("sabre", "trivial"),
        n_seeds=1,
        measure_fidelity=False,
        verbose=False,
    )
    assert len(results) == 4


def test_run_qiskit_sweep_median_seed_is_minus_one(aer):
    # Aggregated results have seed=-1 as a marker.
    spec = make_circuit("qft", n=4)
    results = run_qiskit_sweep(
        [spec], aer,
        opt_levels=(1,),
        layout_methods=("sabre",),
        n_seeds=2,
        measure_fidelity=False,
        verbose=False,
    )
    assert results[0].seed == -1


def test_run_qiskit_sweep_fields_match_spec(aer):
    spec = make_circuit("bv", n=4)
    results = run_qiskit_sweep(
        [spec], aer,
        opt_levels=(1,),
        layout_methods=("sabre",),
        n_seeds=1,
        measure_fidelity=False,
        verbose=False,
    )
    r = results[0]
    assert r.spec_name == spec.name
    assert r.family == "bv"
    assert r.n_qubits == 4
    assert r.compiler == "qiskit"


# --- save_results / load_results roundtrip ---

def test_save_results_creates_json_and_csv(aer, tmp_path):
    spec = make_circuit("clifford", n=3)
    results = run_qiskit_sweep(
        [spec], aer,
        opt_levels=(0,),
        layout_methods=("sabre",),
        n_seeds=1,
        measure_fidelity=False,
        verbose=False,
    )
    paths = save_results(results, tmp_path)
    assert paths["json"].exists()
    assert paths["csv"].exists()


def test_load_results_json_roundtrip(aer, tmp_path):
    spec = make_circuit("qaoa", n=4, graph="complete")
    results = run_qiskit_sweep(
        [spec], aer,
        opt_levels=(1,),
        layout_methods=("sabre",),
        n_seeds=1,
        measure_fidelity=False,
        verbose=False,
    )
    paths = save_results(results, tmp_path)
    loaded = load_results(paths["json"])
    assert len(loaded) == len(results)
    r0, l0 = results[0], loaded[0]
    assert r0.spec_name == l0.spec_name
    assert r0.two_q_before == l0.two_q_before
    assert r0.two_q_after == l0.two_q_after
    assert r0.family == l0.family


def test_load_results_csv_roundtrip(aer, tmp_path):
    spec = make_circuit("qft", n=3)
    results = run_qiskit_sweep(
        [spec], aer,
        opt_levels=(0,),
        layout_methods=("sabre",),
        n_seeds=1,
        measure_fidelity=False,
        verbose=False,
    )
    paths = save_results(results, tmp_path)
    loaded = load_results(paths["csv"])
    assert len(loaded) == len(results)
    r0, l0 = results[0], loaded[0]
    assert r0.n_qubits == l0.n_qubits
    assert abs(r0.routing_overhead - l0.routing_overhead) < 1e-9
    assert abs(r0.compile_s - l0.compile_s) < 1e-9


def test_load_results_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "results.xyz"
    bad.write_text("garbage")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_results(bad)
