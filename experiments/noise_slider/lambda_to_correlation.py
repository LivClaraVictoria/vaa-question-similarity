"""
Diagnostic: map each λ value to the empirical Pearson correlation
r(source_voter_answers, perturbed_clone_voter_answers).

Answers a key question: does the inverse-distance noise model actually
reach "approximate clone territory" (r < 0.775) anywhere in [0, 1]?

Usage:
    python -m experiments.noise_slider.lambda_to_correlation \
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a04.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import _get_question_text_col
from experiments.noise_slider._perturb import admissible_set, perturb_column
from experiments.noise_slider.robustness import LAMBDA_GRID, _derive_seed
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset


N_SEEDS = 10

APPROX_CLONE_R = 0.775          # max |r| among top-5 approximate clones (Exp 3)
APPROX_CLONE_ARCCOS = 0.684     # arccos(0.775) — min non-clone distance in correlation space


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Map λ → empirical Pearson r(source, perturbed_clone) for noise slider."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--lambdas", type=str, default=None,
        help="Comma-separated λ values (default: plan grid)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=N_SEEDS,
        help=f"Seeds per λ for averaging (default: {N_SEEDS})",
    )
    parser.add_argument(
        "--out", type=str,
        default="experiment_results/noise_slider/lambda_to_correlation.csv",
        help="Output CSV path",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(Path(args.config))

    lambda_grid = (
        sorted([float(s.strip()) for s in args.lambdas.split(",")])
        if args.lambdas else LAMBDA_GRID
    )
    n_seeds = args.n_seeds

    dataset = load_dataset(config)
    voters = dataset["voters"]
    candidates = dataset["candidates"]
    questions = dataset["questions"]

    text_col = _get_question_text_col(questions)
    question_ids = sorted(
        questions.loc[questions["ID_question"] < 9_000_000, "ID_question"].tolist()
    )
    print(f"Questions: {len(question_ids)}  |  λ grid: {lambda_grid}  |  Seeds: {n_seeds}")

    rows = []
    for q_id in question_ids:
        src_col = f"answer_{q_id}"
        if src_col not in voters.columns:
            continue

        adm = admissible_set(voters[src_col], candidates[src_col])
        src = voters[src_col].dropna().to_numpy()
        if src.size < 2 or adm.size <= 1:
            continue

        for lam in lambda_grid:
            if lam == 0.0:
                rows.append({"question_id": q_id, "lambda": 0.0, "r_mean": 1.0})
                continue

            rs = []
            for seed_idx in range(n_seeds):
                rng = np.random.default_rng(_derive_seed(q_id, lam, seed_idx, "voters"))
                perturbed, _ = perturb_column(src, lam, adm, rng)
                r = float(np.corrcoef(src, perturbed)[0, 1])
                rs.append(r)
            rows.append({"question_id": q_id, "lambda": lam, "r_mean": float(np.mean(rs))})

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("lambda")["r_mean"]
        .mean()
        .reset_index()
        .rename(columns={"r_mean": "r"})
        .sort_values("lambda")
        .reset_index(drop=True)
    )
    agg["arccos_r"] = np.arccos(np.clip(agg["r"], -1, 1))

    # Print table
    print("\n" + "=" * 65)
    print("λ → empirical Pearson r(source, perturbed_clone) — voter answers")
    print("=" * 65)
    print(f"{'λ':>6}  {'r_mean':>8}  {'arccos(r)':>10}  note")
    print("-" * 65)
    for _, row in agg.iterrows():
        note = ""
        if row["arccos_r"] >= 1.0:
            note = "← above CRW α=1.0 detection limit"
        elif row["arccos_r"] >= APPROX_CLONE_ARCCOS:
            note = "← above approx-clone threshold"
        print(f"{row['lambda']:>6.2f}  {row['r']:>8.4f}  {row['arccos_r']:>10.4f}  {note}")
    print("=" * 65)
    print(f"\nReference lines:")
    print(f"  Approximate-clone threshold  : r={APPROX_CLONE_R:.3f}  arccos={APPROX_CLONE_ARCCOS:.3f}")
    print(f"  CRW α=1.0 detection limit    : r={np.cos(1.0):.3f}  arccos=1.000")
    print(f"  (CRW integrates [0, α]; clones at arccos(r) > α are undetectable)")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"\n→ Saved: {out_path}")


if __name__ == "__main__":
    main()
