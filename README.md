# SuperstaQ Cross-Compiler Benchmark

Compiles the same circuits through two compilers, SuperstaQ (Infleqtion's neutral-atom SDK) and Qiskit's transpiler, and compares gate count, routing overhead, and compile time across hardware topologies.

The question: when does SuperstaQ's neutral-atom hardware (all-to-all connectivity via atom shuttling) beat Qiskit's heavy-hex transpiler (fixed adjacency, SWAP insertion required for non-adjacent gates)?

Circuit generators, the Qiskit baseline, the SuperstaQ cross-compiler harness, and result analysis are all built and tested.

## Why these circuits

Neutral-atom platforms like Infleqtion's Sqale can rearrange qubits mid-circuit (atom shuttling), so long-range entanglement is cheap. On heavy-hex, long-range two-qubit gates need chains of inserted SWAPs. The four circuit families are chosen to make that tradeoff visible:

- **QAOA on varying connectivity**: line, ring, complete graph. Complete-graph QAOA on 8 qubits has 28 two-qubit gates, all non-local. Expensive on heavy-hex, cheap on neutral-atom.
- **QFT with approximation degree control**: O(n^2) controlled-phase gates spanning the full qubit range. Good for isolating long-range connectivity advantage.
- **Bernstein-Vazirani with variable secret**: oracle CX count equals the Hamming weight of the secret, so sweeping secret weight controls how much long-range entanglement the oracle needs.
- **Random Clifford with depth control**: bounded-depth circuits for stress-testing routing without circuit size growing as O(n^2).

## What's in here

| Module | Purpose |
| --- | --- |
| `scxb/qaoa.py` | QAOA circuits on 5 graph families parameterized by n and connectivity |
| `scxb/qft.py` | QFT with approximation degree and decomposition control |
| `scxb/bv.py` | Bernstein-Vazirani with variable oracle sparsity |
| `scxb/clifford.py` | Random Clifford, full or depth-bounded |
| `scxb/spec.py` | `CircuitSpec` dataclass, `REGISTRY`, `make_circuit` dispatcher |
| `tests/` | Circuit structure, edge counts, error paths |

## Install

```
cd superstaq-cross-compiler-benchmark
pip install -e ".[test]"
```

Needs `qiskit + qiskit-aer + numpy + networkx + matplotlib + pandas`. SuperstaQ SDK is optional:

```
pip install -e ".[superstaq]"   # includes qiskit-superstaq
```

Set `SUPERSTAQ_API_KEY` before running the cross-compiler scripts.

## Quick example

```python
from scxb import make_circuit, available_families

print(available_families())  # ['qaoa', 'qft', 'bv', 'clifford']

# QAOA on a complete graph, worst case for fixed topologies
spec = make_circuit("qaoa", n=6, graph="complete", p=1)
print(spec.circuit.count_ops())  # {'h': 6, 'rzz': 15, 'rx': 6}

# QFT, block vs decomposed
spec_block = make_circuit("qft", n=4, decomposed=False)
spec_elem  = make_circuit("qft", n=4, decomposed=True)

# BV with a sparse oracle (2 CX gates)
spec = make_circuit("bv", n=5, secret=0b0011)
print(spec.circuit.count_ops().get("cx"))  # 2

# Clifford with controlled depth
spec = make_circuit("clifford", n=4, depth=20)
```

## Test

```
pytest
```

Covers exact edge counts for line, ring, and complete QAOA graphs; RZZ count = edges x p for deterministic topologies; QFT gate reduction under approximation and block vs decomposed distinction; BV CX count = Hamming weight of secret plus valid-range enforcement; Clifford full-random (seed-reproducible) and depth-controlled mode.

## Run the cross-compiler benchmark

```
# Both compilers, requires SUPERSTAQ_API_KEY
python -m examples.run_cross_compiler

# Qiskit-only baseline, no API key needed
python -m examples.run_cross_compiler --no-superstaq

# Fast smoke test
python -m examples.run_cross_compiler --n-values 4 --families qaoa \
    --graphs ring --no-plots --quiet
```

Results land in `results/` as `results.json`, `results.csv`, and PNG plots.

## Results

Compiled n in {4, 6, 8} across four circuit families through:

- **Qiskit** opt-3 / SABRE targeting FakeSherbrooke (127-qubit heavy-hex, ECR native gates)
- **SuperstaQ** `cq_sqale_simulator` (Infleqtion Sqale neutral-atom, CZ native gates, all-to-all via atom shuttling)

**Coverage:** the SuperstaQ free-tier account ran out of credits partway through the sweep. Qiskit covers all 45 circuits across n = {4, 6, 8}. SuperstaQ covers all 15 circuits at n=4 and three QAOA circuits at n=6 (line, ring, complete). No SuperstaQ data at n=8.

### The neutral-atom advantage is topology-conditional

SuperstaQ does not universally beat Qiskit. It depends on whether the circuit's connectivity pattern conflicts with heavy-hex. The 27-37% Qiskit-routing-overhead numbers below are from the FakeSherbrooke (127-qubit heavy-hex) run described above; Qiskit's side of every comparison here is specific to that target topology, and would differ against a different coupling map.

| Condition | Winner | Example |
| --- | --- | --- |
| Ring QAOA (n=4) | SuperstaQ, 27% lower | Qiskit 2.75x vs SuperstaQ 2.00x |
| Ring QAOA (n=6) | SuperstaQ, 37% lower | Qiskit 3.17x vs SuperstaQ 2.00x |
| Line QAOA (n=4, 6) | Tie | Both 2.00x |
| ER QAOA (n=4, median of 5 seeds) | Tie | Both 2.00x median |
| Complete-graph QAOA (n=4, 6) | Near-tie, within 7% | Both ~3.0-3.3x |
| QFT (n=4) | Qiskit, 28% lower | Qiskit 2.25x vs SuperstaQ 3.13x |
| BV (n=4) | Tie | Identical gate counts |
| Clifford (n=4, median of 5 seeds) | Qiskit, 18% lower | Qiskit 1.17x vs SuperstaQ 1.43x |

**Qiskit-only, n=8:** with 5 random ER seeds, the Qiskit ER median is 3.47x (IQR 3.21-3.54x), just above complete-graph (3.32x). Reproducible across seeds. Likely reason: SABRE exploits the regular all-pairs structure of complete graphs to find a good layout on heavy-hex; random ER topologies at n=8 don't benefit from that regularity even with fewer edges.

**Compile time:** SuperstaQ is 15-25x slower per circuit (median ~0.7-1.4s vs Qiskit ~0.03-0.08s). Network latency to the SuperstaQ API, not compiler quality.

**Why SuperstaQ loses on QFT and Clifford (n=4):** Qiskit opt-3 does aggressive gate fusion and ECR-native synthesis that compresses QFT's all-pairs controlled-phase structure. SuperstaQ's CZ decomposition of the same structure produces more gates. Gate-synthesis difference, not a routing difference; both compilers have effective all-to-all access at these small sizes.

### Plots

In `results/phase4/`, generated by `scxb/plotting.generate_all`:

- `qaoa_overhead_vs_connectivity.png`: QAOA routing overhead vs graph connectivity per n, Qiskit vs SuperstaQ. Ring advantage visible at n=4 and n=6. Shaded IQR band for ER (5-seed average).
- `overhead_by_family.png`: grouped bars, all circuit types side by side, IQR error bars on seed-averaged families.
- `compile_time_crosscompiler.png`: log-scale compile time per n, ~20x gap.
- `summary_table.csv`: one row per circuit group with winner and advantage_pct, only for (group, n) pairs where both compilers have data.

Earlier Qiskit-only baseline plots are in `results/phase2_baseline/`.

## Methodology

**Single-instance circuits** (one compilation per n): QAOA on line, ring, complete graphs; QFT; Bernstein-Vazirani. Fully determined by n and graph type, so one compilation is exact.

**Seed-averaged circuits** (5 random instances per n): ER QAOA and random Clifford. Sampled randomly, so a single instance can be unrepresentative. The benchmark draws 5 independent instances (seeds 0-4), compiles each, and reports median routing_overhead with 25th/75th percentile error bars.

**Two-layer aggregation for Qiskit:** Qiskit's transpiler is itself randomized, routing depends on the seed passed to `transpile()`. Each circuit instance runs 3 transpiler seeds and takes the median. So a Qiskit ER QAOA point at n=8 has two aggregation layers: 3 transpiler seeds per graph instance, then 5 graph instances, median at each layer. SuperstaQ's compilation is deterministic, so only the graph-instance layer applies: Qiskit's result is the median of up to 15 compile attempts per group (3 x 5), SuperstaQ's is the median of 5.

**Why `routing_overhead`, not `swap_count`:** both compilers report `swap_count = 0`. That looks wrong but isn't.

- Qiskit/FakeSherbrooke: SABRE at opt-3 picks a layout where the circuit's edges map to adjacent heavy-hex ECR links. When routing is needed, the SWAP gets absorbed directly into ECR gate synthesis, so no explicit `swap` gate survives `count_ops()`. The routing cost shows up as extra ECR gates in `two_q_after`.
- SuperstaQ/Sqale: no SWAP insertion, as expected. Overhead comes from CZ decomposition of input RZZ gates (one RZZ becomes 2 CZ gates), not from routing.

`routing_overhead = two_q_after / two_q_before` captures both effects in one comparable number and is the right metric here.

**Data coverage:** SuperstaQ's `cq_sqale_simulator` charges credits per job; the free-tier account ran out partway through the sweep. n=4: all 15 instances compiled. n=6: 3 of 15 (QAOA line, ring, complete; ER and Clifford skipped). n=8: 0 of 15. Qiskit is complete at 45/45. The ring-topology finding holds because ring QAOA is deterministic and was compiled at both n=4 and n=6. ER QAOA and Clifford at n>=6 on the SuperstaQ side would need more credits to check.

## Roadmap

- **Fidelity:** simulate compiled circuits under realistic noise models per backend, compare output fidelity, not just gate count.
- **Larger n:** extend to n in {10, 12} to see if the ring-topology advantage survives or caps out (n=8 ring is already a tie for the Qiskit-only data).

## References

- Infleqtion SuperstaQ: https://superstaq.infleqtion.com
- Harrigan et al., "Quantum approximate optimization of non-planar graph problems on a planar superconducting processor," Nature Physics 2021 (connectivity cost).
- Li, Ding, Xie, "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices" (SABRE), ASPLOS 2019.
