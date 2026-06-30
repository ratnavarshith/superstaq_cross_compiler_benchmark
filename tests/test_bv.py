"""Tests for make_bv.

Key invariant: CX count in the oracle = Hamming weight of the secret
bitstring. This is what the tests pin down — it's the routing-difficulty
knob that the existing quantum-compiler-bench BV implementation doesn't vary.
"""
import pytest
from scxb.bv import make_bv


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_bv_qubit_count(n):
    qc = make_bv(n)
    assert qc.num_qubits == n


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_bv_classical_bit_count(n):
    # Output register has n-1 bits (one per input qubit).
    qc = make_bv(n)
    assert qc.num_clbits == n - 1


def test_bv_all_ones_cx_count():
    # all_ones secret for n=5: 4 input bits, all set → 4 oracle CX gates.
    qc = make_bv(5, secret="all_ones")
    assert qc.count_ops().get("cx", 0) == 4


def test_bv_default_is_all_ones():
    qc_default = make_bv(5)
    qc_all_ones = make_bv(5, secret="all_ones")
    assert qc_default.count_ops() == qc_all_ones.count_ops()


def test_bv_secret_zero_produces_no_oracle_cx():
    # Secret = 0: no bits set → empty oracle, no CX gates at all.
    qc = make_bv(4, secret=0)
    assert qc.count_ops().get("cx", 0) == 0


def test_bv_cx_count_equals_hamming_weight():
    # secret=0b1010 = 10 over 4 input bits: popcount = 2
    qc = make_bv(5, secret=0b1010)
    assert qc.count_ops().get("cx", 0) == 2


@pytest.mark.parametrize("hw", [1, 2, 3])
def test_bv_cx_count_for_various_hamming_weights(hw):
    # Construct secret with exactly `hw` bits set in a 4-bit field (n=5).
    secret = (1 << hw) - 1  # e.g. hw=2 → 0b11
    qc = make_bv(5, secret=secret)
    assert qc.count_ops().get("cx", 0) == hw


def test_bv_random_secret_is_nonzero():
    # Random mode always produces at least one oracle CX.
    for seed in range(6):
        qc = make_bv(5, secret="random", seed=seed)
        assert qc.count_ops().get("cx", 0) >= 1


def test_bv_random_secret_reproducible():
    qc1 = make_bv(5, secret="random", seed=7)
    qc2 = make_bv(5, secret="random", seed=7)
    assert qc1.count_ops() == qc2.count_ops()


def test_bv_has_measurements():
    qc = make_bv(4)
    assert qc.num_clbits > 0


def test_bv_out_of_range_secret_raises():
    with pytest.raises(ValueError, match="out of range"):
        make_bv(4, secret=100)  # max valid secret for n=4 is 6 (0b110)


def test_bv_wrong_secret_type_raises():
    with pytest.raises(TypeError):
        make_bv(4, secret=3.14)
