"""Unit tests for experiments.noise_slider._perturb."""

import numpy as np
import pandas as pd
import pytest

from experiments.noise_slider._perturb import admissible_set, perturb_column


ADM_4 = np.array([0.0, 25.0, 75.0, 100.0])
ADM_5 = np.array([0.0, 25.0, 50.0, 75.0, 100.0])


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_admissible_set_discovers_4_point_scale():
    voters = pd.Series([0, 25, 75, 100, 0, 75, np.nan])
    cands = pd.Series([25, 100, np.nan, 0, 0, 75])
    assert np.array_equal(admissible_set(voters, cands), ADM_4)


def test_admissible_set_discovers_5_point_scale():
    voters = pd.Series([0, 25, 50, 75, 100])
    assert np.array_equal(admissible_set(voters), ADM_5)


def test_lam_zero_is_identity():
    x = np.tile(ADM_4, 250)  # length 1000
    out, tel = perturb_column(x, lam=0.0, admissible=ADM_4, rng=_make_rng(0))
    assert np.array_equal(out, x)
    assert tel["n_switched"] == 0
    assert tel["n_kept"] == x.size


def test_lam_one_switches_every_entry():
    x = np.tile(ADM_4, 250)
    out, tel = perturb_column(x, lam=1.0, admissible=ADM_4, rng=_make_rng(1))
    # every entry must differ from its starting value
    assert np.all(out != x)
    assert tel["n_switched"] == x.size
    # output must stay within admissible
    assert np.all(np.isin(out, ADM_4))


def test_lam_half_approximately_half_switched():
    x = np.tile(ADM_4, 2500)  # length 10k
    out, tel = perturb_column(x, lam=0.5, admissible=ADM_4, rng=_make_rng(2))
    switch_rate = tel["n_switched"] / x.size
    assert abs(switch_rate - 0.5) < 0.02   # well within tolerance for n=10k


def test_inverse_distance_transition_ratio_from_25():
    # From x=25 with admissible {0,25,75,100}:
    #   P(→0) ∝ 1/25, P(→75) ∝ 1/50, P(→100) ∝ 1/75
    # Ratio P(→0) / P(→75) should be 2. Sample empirically.
    n = 200_000
    x = np.full(n, 25.0)
    out, _ = perturb_column(x, lam=1.0, admissible=ADM_4, rng=_make_rng(3))
    to_zero = int(np.sum(out == 0))
    to_75 = int(np.sum(out == 75))
    to_100 = int(np.sum(out == 100))
    # Ratio P(→0)/P(→75) ≈ 2
    assert abs(to_zero / to_75 - 2.0) < 0.05
    # Ratio P(→75)/P(→100) ≈ 1.5
    assert abs(to_75 / to_100 - 1.5) < 0.05
    # No self-transitions
    assert int(np.sum(out == 25)) == 0


def test_nan_stays_nan():
    x = np.array([25.0, np.nan, 75.0, np.nan, 0.0])
    out, tel = perturb_column(x, lam=1.0, admissible=ADM_4, rng=_make_rng(4))
    assert np.isnan(out[1]) and np.isnan(out[3])
    assert tel["n_nan"] == 2
    assert tel["n_switched"] == 3


def test_output_stays_in_admissible():
    rng = _make_rng(5)
    x = np.tile(ADM_5, 200)
    out, _ = perturb_column(x, lam=0.7, admissible=ADM_5, rng=rng)
    non_nan = out[~np.isnan(out)]
    assert np.all(np.isin(non_nan, ADM_5))


def test_trivial_admissible_set_is_no_op():
    x = np.full(100, 50.0)
    out, tel = perturb_column(x, lam=1.0, admissible=np.array([50.0]), rng=_make_rng(6))
    assert np.array_equal(out, x)
    assert tel["n_skipped_small"] == 100


def test_value_outside_admissible_is_skipped():
    # x contains 37, which is not in admissible — that entry should be left untouched.
    x = np.array([25.0, 37.0, 75.0])
    out, tel = perturb_column(x, lam=1.0, admissible=ADM_4, rng=_make_rng(7))
    assert out[1] == 37.0
    assert tel["n_skipped_offset"] == 1
    # other two must have switched
    assert out[0] != 25.0 and out[2] != 75.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
