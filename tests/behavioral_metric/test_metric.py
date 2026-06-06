"""Unit tests for the behavioral L1 distance metric and the shared answer-based base class."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vqs.similarity_metrics import (
    BehavioralL1DistanceCalculator,
    CorrelationDistanceCalculator,
    get_calculator,
)

COLS = ["answer_1", "answer_2"]


def _cfg(dist="BEHAVIORAL-L1"):
    return SimpleNamespace(
        DISTANCE_HASH_PARAMS=[], correlation_answer_source="both", dist=dist
    )


def _beh():
    return BehavioralL1DistanceCalculator(_cfg())


def _d(df):
    return _beh()._distance_matrix(df, COLS)[0, 1]


def test_identical_columns_zero():
    a = np.tile([0.0, 100.0], 20)  # 40 rows
    assert _d(pd.DataFrame({"answer_1": a, "answer_2": a.copy()})) == 0.0


def test_exact_negation_zero():
    a = np.tile([0.0, 100.0], 20)
    assert _d(pd.DataFrame({"answer_1": a, "answer_2": 100.0 - a})) < 1e-12


def test_unrelated_mean_abs_diff():
    # a cycles 0/50/100, b constant 50 -> mean|a-50| = 100/3 -> /100 = 0.3333
    a = np.tile([0.0, 50.0, 100.0], 13)  # 39 rows
    b = np.full(39, 50.0)
    assert _d(pd.DataFrame({"answer_1": a, "answer_2": b})) == pytest.approx(1 / 3, abs=1e-6)


def test_negation_branch_wins_for_anticorrelation():
    # constant 0 vs constant 100: d_id=1.0, but negation makes them identical -> 0.0
    df = pd.DataFrame({"answer_1": np.zeros(40), "answer_2": np.full(40, 100.0)})
    assert _d(df) == 0.0


def test_pairwise_nan_deletion():
    # 40 rows; identical where both present, but a has NaNs -> still distance 0 over overlap
    a = np.tile([0.0, 100.0], 20)
    b = a.copy()
    a[:5] = np.nan  # 35 overlapping rows (>= MIN_OVERLAP)
    assert _d(pd.DataFrame({"answer_1": a, "answer_2": b})) == 0.0


def test_insufficient_overlap_is_max():
    # only 10 rows where both are non-NaN (< MIN_OVERLAP=30) -> distance 1.0
    a = np.full(40, np.nan)
    b = np.full(40, np.nan)
    a[:10] = 0.0
    b[:10] = 0.0
    assert _d(pd.DataFrame({"answer_1": a, "answer_2": b})) == 1.0


def test_normalized_range():
    rng = np.random.default_rng(0)
    a = rng.choice([0.0, 25.0, 50.0, 75.0, 100.0], size=60)
    b = rng.choice([0.0, 25.0, 50.0, 75.0, 100.0], size=60)
    d = _d(pd.DataFrame({"answer_1": a, "answer_2": b}))
    assert 0.0 <= d <= 1.0


def test_correlation_base_still_works():
    # Regression: the refactored correlation calculator shares the new base class.
    a = np.tile([0.0, 100.0], 20)
    df = pd.DataFrame({"answer_1": a, "answer_2": a.copy()})
    calc = CorrelationDistanceCalculator(_cfg("ANSWER-CORRELATION"))
    # identical, perfectly correlated -> 1 - |r| = 0
    assert calc._distance_matrix(df, COLS)[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_registry_wiring():
    calc = get_calculator(_cfg("BEHAVIORAL-L1"))
    assert isinstance(calc, BehavioralL1DistanceCalculator)


def test_cache_params_include_split_keys():
    # The answer-based cache hash must distinguish source + voter split.
    calc = _beh()
    for key in ("correlation_answer_source", "train_voter_fraction", "split_seed"):
        assert key in calc.important_params_list
