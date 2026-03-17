"""Test script for the ANSWER-CORRELATION distance metric.

Uses synthetic data — no real data files or GPU needed.
Run: python -m scripts.test_answer_correlation_metric
"""

import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import shutil

from vqs.similarity_metrics import CorrelationDistanceCalculator, ArcCosCorrelationDistanceCalculator, METRIC_REGISTRY


def make_config(tmp_dir: Path, answer_source: str = "voters") -> SimpleNamespace:
    return SimpleNamespace(
        data_year=2023,
        dist="ANSWER-CORRELATION",
        data_choice="cleaned",
        clone_id=None,
        embedding_instruction=None,
        embedding_task=None,
        correlation_answer_source=answer_source,
        DISTANCE_CACHE_DIR=tmp_dir / "cache",
        DISTANCE_HASH_PARAMS=[
            "data_year", "dist", "data_choice", "clone_id",
            "embedding_instruction", "embedding_task", "correlation_answer_source",
        ],
        results_file_type="csv",
        save_results=True,
        load_voters=True,
        load_candidates=True,
    )


def make_dataset(n_voters: int = 200, n_candidates: int = 50) -> dict:
    """Create synthetic dataset with 5 questions and known correlation structure."""
    np.random.seed(42)
    question_ids = [10001, 10002, 10003, 10004, 10005]
    question_texts = [
        "Should taxes be increased?",
        "Should taxes be raised?",           # near-duplicate of Q1
        "Should immigration be restricted?",
        "Should the military budget grow?",
        "Should education funding increase?",
    ]

    # Q1 and Q2 are highly correlated (near-identical questions)
    # Q3, Q4, Q5 are independent
    def generate_answers(n: int) -> pd.DataFrame:
        base = np.random.randint(0, 101, size=(n, 5)).astype(float)
        # Make Q2 a noisy copy of Q1 (high positive correlation)
        base[:, 1] = base[:, 0] + np.random.normal(0, 5, n)
        base[:, 1] = np.clip(base[:, 1], 0, 100)
        cols = {f"answer_{qid}": base[:, i] for i, qid in enumerate(question_ids)}
        return pd.DataFrame(cols)

    df_voters = generate_answers(n_voters)
    df_candidates = generate_answers(n_candidates)

    df_questions = pd.DataFrame({
        "ID_question": question_ids,
        "question_EN": question_texts,
    })

    return {
        "questions": df_questions,
        "voters": df_voters,
        "candidates": df_candidates,
    }


def test_registry():
    """Test that ANSWER-CORRELATION is registered."""
    print("--- Test: Registry ---")
    assert "ANSWER-CORRELATION" in METRIC_REGISTRY, "Missing from METRIC_REGISTRY"
    entry = METRIC_REGISTRY["ANSWER-CORRELATION"]
    assert entry["class"] is CorrelationDistanceCalculator
    print("PASS: ANSWER-CORRELATION is registered correctly.")


def test_basic_computation(tmp_dir: Path):
    """Test distance computation with synthetic data."""
    print("\n--- Test: Basic Computation ---")
    config = make_config(tmp_dir / "basic")
    dataset = make_dataset()

    calc = CorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    expected_cols = {"Qu1", "Qu2", "ID1", "ID2", "Distance", "Type"}
    assert set(result.columns) == expected_cols, f"Columns: {result.columns.tolist()}"

    n_questions = 5
    expected_rows = n_questions * (n_questions - 1) // 2  # C(5,2) = 10
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"

    # All distances in [0, 1]
    assert (result["Distance"] >= 0).all(), "Found negative distances"
    assert (result["Distance"] <= 1.0 + 1e-9).all(), "Found distances > 1"

    # All types should be Real-Symmetric
    assert (result["Type"] == "Real-Symmetric").all()

    print(f"PASS: Output shape ({len(result)} rows, {len(result.columns)} cols) is correct.")
    print(f"PASS: Distance range [{result['Distance'].min():.4f}, {result['Distance'].max():.4f}] is in [0, 1].")

    return result


def test_correlated_questions_closer(result: pd.DataFrame):
    """Test that Q1-Q2 (near-duplicate) has smaller distance than other pairs."""
    print("\n--- Test: Correlated Questions Are Closer ---")
    q1_q2 = result[(result["ID1"] == 10001) & (result["ID2"] == 10002)]
    assert len(q1_q2) == 1
    d_correlated = q1_q2["Distance"].iloc[0]

    other_pairs = result[~((result["ID1"] == 10001) & (result["ID2"] == 10002))]
    d_others_mean = other_pairs["Distance"].mean()

    print(f"  Q1-Q2 distance (correlated):    {d_correlated:.4f}")
    print(f"  Other pairs mean distance:       {d_others_mean:.4f}")

    assert d_correlated < d_others_mean, (
        f"Correlated pair ({d_correlated:.4f}) should be closer than average ({d_others_mean:.4f})"
    )
    print("PASS: Correlated questions have smaller distance.")


def test_symmetry(tmp_dir: Path):
    """Test that the metric is symmetric (order of IDs doesn't matter)."""
    print("\n--- Test: Symmetry ---")
    config = make_config(tmp_dir / "symmetry")
    dataset = make_dataset()

    calc = CorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)

    # Rebuild correlation matrix from edge list and verify symmetry
    ids = sorted(set(result["ID1"]) | set(result["ID2"]))
    n = len(ids)
    id_to_idx = {qid: i for i, qid in enumerate(ids)}
    matrix = np.zeros((n, n))
    for _, row in result.iterrows():
        u, v = id_to_idx[row["ID1"]], id_to_idx[row["ID2"]]
        matrix[u, v] = row["Distance"]
        matrix[v, u] = row["Distance"]

    assert np.allclose(matrix, matrix.T), "Distance matrix is not symmetric"
    print("PASS: Distance matrix is symmetric.")


def test_identical_answers_zero_distance(tmp_dir: Path):
    """Test that questions with identical answers get distance = 0."""
    print("\n--- Test: Identical Answers → Distance 0 ---")
    config = make_config(tmp_dir / "identical")

    np.random.seed(99)
    n = 100
    answers = np.random.randint(0, 101, size=n).astype(float)
    df_voters = pd.DataFrame({
        "answer_1": answers,
        "answer_2": answers,  # identical to Q1
        "answer_3": np.random.randint(0, 101, size=n).astype(float),
    })
    df_questions = pd.DataFrame({
        "ID_question": [1, 2, 3],
        "question_EN": ["Q1", "Q2 (clone of Q1)", "Q3 (independent)"],
    })
    dataset = {"questions": df_questions, "voters": df_voters}

    calc = CorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)

    d_q1_q2 = result[(result["ID1"] == 1) & (result["ID2"] == 2)]["Distance"].iloc[0]
    assert abs(d_q1_q2) < 1e-10, f"Identical answers should have distance 0, got {d_q1_q2}"
    print(f"PASS: Identical answers → distance = {d_q1_q2:.1e}")


def test_negated_answers_zero_distance(tmp_dir: Path):
    """Test that negated answers (100 - x) also get distance = 0."""
    print("\n--- Test: Negated Answers → Distance 0 ---")
    config = make_config(tmp_dir / "negated")

    np.random.seed(99)
    n = 100
    answers = np.random.randint(0, 101, size=n).astype(float)
    df_voters = pd.DataFrame({
        "answer_1": answers,
        "answer_2": 100.0 - answers,  # perfectly anti-correlated
        "answer_3": np.random.randint(0, 101, size=n).astype(float),
    })
    df_questions = pd.DataFrame({
        "ID_question": [1, 2, 3],
        "question_EN": ["Q1", "Q2 (negation of Q1)", "Q3 (independent)"],
    })
    dataset = {"questions": df_questions, "voters": df_voters}

    calc = CorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)

    d_q1_q2 = result[(result["ID1"] == 1) & (result["ID2"] == 2)]["Distance"].iloc[0]
    assert abs(d_q1_q2) < 1e-10, f"Negated answers should have distance 0, got {d_q1_q2}"
    print(f"PASS: Negated answers → distance = {d_q1_q2:.1e}")


def test_answer_source_options(tmp_dir: Path):
    """Test that different answer sources produce different results."""
    print("\n--- Test: Answer Source Options ---")
    dataset = make_dataset(n_voters=200, n_candidates=50)

    results = {}
    for source in ("voters", "candidates", "both"):
        config = make_config(tmp_dir / f"source_{source}", answer_source=source)
        calc = CorrelationDistanceCalculator(config)
        results[source] = calc.calculate_distance(dataset, config)
        print(f"  {source}: mean distance = {results[source]['Distance'].mean():.4f}")

    # Results should differ between sources (different respondent pools)
    d_voters = results["voters"]["Distance"].values
    d_candidates = results["candidates"]["Distance"].values
    assert not np.allclose(d_voters, d_candidates, atol=0.01), (
        "Voters and candidates should produce different distances"
    )
    print("PASS: Different answer sources produce different distance values.")


def test_fake_data_rejected(tmp_dir: Path):
    """Test that fake data raises ValueError."""
    print("\n--- Test: Fake Data Rejected ---")
    config = make_config(tmp_dir / "fake")
    config.data_choice = "fake"
    dataset = make_dataset()

    calc = CorrelationDistanceCalculator(config)
    try:
        calc.calculate_distance(dataset, config)
        print("FAIL: Should have raised ValueError for fake data")
    except ValueError as e:
        print(f"PASS: Correctly rejected fake data: {e}")


def test_missing_voters_rejected(tmp_dir: Path):
    """Test that missing voter data raises ValueError."""
    print("\n--- Test: Missing Voters Rejected ---")
    config = make_config(tmp_dir / "missing")
    dataset = make_dataset()
    del dataset["voters"]

    calc = CorrelationDistanceCalculator(config)
    try:
        calc.calculate_distance(dataset, config)
        print("FAIL: Should have raised ValueError for missing voters")
    except ValueError as e:
        print(f"PASS: Correctly rejected missing voters: {e}")


def test_caching(tmp_dir: Path):
    """Test that results are cached and reused."""
    print("\n--- Test: Caching ---")
    config = make_config(tmp_dir / "caching")
    dataset = make_dataset()

    calc = CorrelationDistanceCalculator(config)

    # First call: computes
    result1 = calc.calculate_distance(dataset, config)

    # Second call: should load from cache
    result2 = calc.calculate_distance(dataset, config)

    # Use approximate comparison since CSV round-tripping loses float precision
    assert result1.shape == result2.shape, "Cached result has different shape"
    assert list(result1.columns) == list(result2.columns), "Cached result has different columns"
    assert np.allclose(result1["Distance"].values, result2["Distance"].values, atol=1e-10), (
        "Cached Distance values differ beyond tolerance"
    )
    assert (result1["ID1"].values == result2["ID1"].values).all(), "ID1 mismatch"
    assert (result1["ID2"].values == result2["ID2"].values).all(), "ID2 mismatch"
    print("PASS: Caching works correctly.")


def make_arccos_config(tmp_dir: Path, answer_source: str = "voters") -> SimpleNamespace:
    config = make_config(tmp_dir, answer_source)
    config.dist = "ANSWER-CORRELATION-ARCCOS"
    return config


def test_arccos_registry():
    """Test that ANSWER-CORRELATION-ARCCOS is registered."""
    print("\n--- Test: ArcCos Registry ---")
    assert "ANSWER-CORRELATION-ARCCOS" in METRIC_REGISTRY, "Missing from METRIC_REGISTRY"
    entry = METRIC_REGISTRY["ANSWER-CORRELATION-ARCCOS"]
    assert entry["class"] is ArcCosCorrelationDistanceCalculator
    print("PASS: ANSWER-CORRELATION-ARCCOS is registered correctly.")


def test_arccos_basic_computation(tmp_dir: Path):
    """Test arccos distance computation with synthetic data."""
    print("\n--- Test: ArcCos Basic Computation ---")
    config = make_arccos_config(tmp_dir / "arccos_basic")
    dataset = make_dataset()

    calc = ArcCosCorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)

    # Check output format
    assert isinstance(result, pd.DataFrame)
    n_questions = 5
    expected_rows = n_questions * (n_questions - 1) // 2
    assert len(result) == expected_rows

    # All distances in [0, pi/2]
    assert (result["Distance"] >= 0).all(), "Found negative distances"
    assert (result["Distance"] <= np.pi / 2 + 1e-9).all(), f"Found distances > pi/2: max={result['Distance'].max()}"

    print(f"PASS: Distance range [{result['Distance'].min():.4f}, {result['Distance'].max():.4f}] is in [0, pi/2].")
    return result


def test_arccos_identical_zero(tmp_dir: Path):
    """Test that identical answers give distance 0 with arccos metric."""
    print("\n--- Test: ArcCos Identical → 0 ---")
    config = make_arccos_config(tmp_dir / "arccos_identical")
    np.random.seed(99)
    n = 100
    answers = np.random.randint(0, 101, size=n).astype(float)
    df_voters = pd.DataFrame({
        "answer_1": answers,
        "answer_2": answers,
        "answer_3": np.random.randint(0, 101, size=n).astype(float),
    })
    df_questions = pd.DataFrame({
        "ID_question": [1, 2, 3],
        "question_EN": ["Q1", "Q2 (clone)", "Q3"],
    })
    dataset = {"questions": df_questions, "voters": df_voters}

    calc = ArcCosCorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)
    d = result[(result["ID1"] == 1) & (result["ID2"] == 2)]["Distance"].iloc[0]
    assert abs(d) < 1e-10, f"Identical answers should have distance 0, got {d}"
    print(f"PASS: Identical answers → distance = {d:.1e}")


def test_arccos_negated_zero(tmp_dir: Path):
    """Test that negated answers give distance 0 with arccos metric."""
    print("\n--- Test: ArcCos Negated → 0 ---")
    config = make_arccos_config(tmp_dir / "arccos_negated")
    np.random.seed(99)
    n = 100
    answers = np.random.randint(0, 101, size=n).astype(float)
    df_voters = pd.DataFrame({
        "answer_1": answers,
        "answer_2": 100.0 - answers,
        "answer_3": np.random.randint(0, 101, size=n).astype(float),
    })
    df_questions = pd.DataFrame({
        "ID_question": [1, 2, 3],
        "question_EN": ["Q1", "Q2 (negation)", "Q3"],
    })
    dataset = {"questions": df_questions, "voters": df_voters}

    calc = ArcCosCorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)
    d = result[(result["ID1"] == 1) & (result["ID2"] == 2)]["Distance"].iloc[0]
    assert abs(d) < 1e-10, f"Negated answers should have distance 0, got {d}"
    print(f"PASS: Negated answers → distance = {d:.1e}")


def test_arccos_uncorrelated_max(tmp_dir: Path):
    """Test that uncorrelated answers give distance ~pi/2."""
    print("\n--- Test: ArcCos Uncorrelated → pi/2 ---")
    config = make_arccos_config(tmp_dir / "arccos_uncorrelated")
    np.random.seed(42)
    n = 10000  # large N for stable correlation near 0
    df_voters = pd.DataFrame({
        "answer_1": np.random.randint(0, 101, size=n).astype(float),
        "answer_2": np.random.randint(0, 101, size=n).astype(float),
    })
    df_questions = pd.DataFrame({
        "ID_question": [1, 2],
        "question_EN": ["Q1", "Q2"],
    })
    dataset = {"questions": df_questions, "voters": df_voters}

    calc = ArcCosCorrelationDistanceCalculator(config)
    result = calc.calculate_distance(dataset, config)
    d = result["Distance"].iloc[0]
    assert abs(d - np.pi / 2) < 0.1, f"Uncorrelated should be near pi/2={np.pi/2:.4f}, got {d:.4f}"
    print(f"PASS: Uncorrelated → distance = {d:.4f} (pi/2 = {np.pi/2:.4f})")


def test_arccos_rank_order_matches_old(tmp_dir: Path):
    """Test that arccos preserves the rank order of the old 1-|r| metric."""
    print("\n--- Test: ArcCos Rank Order Matches Old ---")
    dataset = make_dataset()

    old_config = make_config(tmp_dir / "rankorder_old")
    old_calc = CorrelationDistanceCalculator(old_config)
    old_result = old_calc.calculate_distance(dataset, old_config)

    new_config = make_arccos_config(tmp_dir / "rankorder_new")
    new_calc = ArcCosCorrelationDistanceCalculator(new_config)
    new_result = new_calc.calculate_distance(dataset, new_config)

    old_order = old_result.sort_values("Distance").reset_index(drop=True)
    new_order = new_result.sort_values("Distance").reset_index(drop=True)

    assert (old_order["ID1"].values == new_order["ID1"].values).all(), "Rank order differs"
    assert (old_order["ID2"].values == new_order["ID2"].values).all(), "Rank order differs"
    print("PASS: Rank order is identical between 1-|r| and arccos(|r|).")


def main():
    tmp_dir = Path("test_answer_correlation_tmp")
    tmp_dir.mkdir(exist_ok=True)

    passed = 0
    failed = 0

    tests = [
        ("Registry", lambda: test_registry()),
        ("Basic Computation", lambda: test_basic_computation(tmp_dir)),
        ("Correlated Closer", lambda: test_correlated_questions_closer(
            test_basic_computation.__wrapped__(tmp_dir) if hasattr(test_basic_computation, '__wrapped__') else
            CorrelationDistanceCalculator(make_config(tmp_dir / "corr_closer")).calculate_distance(make_dataset(), make_config(tmp_dir / "corr_closer"))
        )),
        ("Symmetry", lambda: test_symmetry(tmp_dir)),
        ("Identical → 0", lambda: test_identical_answers_zero_distance(tmp_dir)),
        ("Negated → 0", lambda: test_negated_answers_zero_distance(tmp_dir)),
        ("Answer Sources", lambda: test_answer_source_options(tmp_dir)),
        ("Fake Rejected", lambda: test_fake_data_rejected(tmp_dir)),
        ("Missing Voters", lambda: test_missing_voters_rejected(tmp_dir)),
        ("Caching", lambda: test_caching(tmp_dir)),
    ]

    # Run tests sequentially (simpler flow)
    try:
        test_registry()
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed += 1

    try:
        result = test_basic_computation(tmp_dir)
        passed += 1
        try:
            test_correlated_questions_closer(result)
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed += 2  # skip dependent test too

    for name, test_fn in [
        ("Symmetry", lambda: test_symmetry(tmp_dir)),
        ("Identical → 0", lambda: test_identical_answers_zero_distance(tmp_dir)),
        ("Negated → 0", lambda: test_negated_answers_zero_distance(tmp_dir)),
        ("Answer Sources", lambda: test_answer_source_options(tmp_dir)),
        ("Fake Rejected", lambda: test_fake_data_rejected(tmp_dir)),
        ("Missing Voters", lambda: test_missing_voters_rejected(tmp_dir)),
        ("Caching", lambda: test_caching(tmp_dir)),
        ("ArcCos Registry", lambda: test_arccos_registry()),
        ("ArcCos Basic", lambda: test_arccos_basic_computation(tmp_dir)),
        ("ArcCos Identical → 0", lambda: test_arccos_identical_zero(tmp_dir)),
        ("ArcCos Negated → 0", lambda: test_arccos_negated_zero(tmp_dir)),
        ("ArcCos Uncorrelated → pi/2", lambda: test_arccos_uncorrelated_max(tmp_dir)),
        ("ArcCos Rank Order", lambda: test_arccos_rank_order_matches_old(tmp_dir)),
    ]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL ({name}): {e}")
            failed += 1

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
