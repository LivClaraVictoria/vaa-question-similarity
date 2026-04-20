"""
Experiment 3 Sanity Check: Mini vs Full Questionnaire

Does CRW on the full 75 questions bring recommendations closer to
what the expert-curated 30-question mini questionnaire produces?

Compares CRW(full) vs CRW(mini) — both sides get CRW treatment.

Uses E5-INSTRUCT at alpha=0.3 (optimal from Experiment 1).

Usage:
    python -m scripts.exp3_sanity_check
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.recommendation_engine import RecommendationEngine
from cross_run_analysis.analyzer import CrossRunAnalyzer
from dependencies import add_candidate_voting_recommendations, SVDataFrame

CONFIG_PATH = Path("configs/base_pipeline/pipeline_e5_instruct_ZH.py")
ALPHA = 0.3
N_JACCARD = 36  # ZH seats in 2023
OUTPUT_DIR = Path("experiment_results/exp3_sanity_check")


def filter_to_mini(dataset: dict) -> dict:
    """Filter dataset to keep only mini (rapide=1) questions."""
    df_questions = dataset["questions"]
    mini_ids = set(df_questions.loc[df_questions["rapide"] == 1, "ID_question"])

    df_q = df_questions[df_questions["ID_question"].isin(mini_ids)].copy()

    keep_answer = {f"answer_{qid}" for qid in mini_ids}
    keep_weight = {f"weight_{qid}" for qid in mini_ids}

    df_v = dataset["voters"].copy()
    drop_v = [c for c in df_v.columns
              if (c.startswith("answer_") and c not in keep_answer)
              or (c.startswith("weight_") and c not in keep_weight)]
    df_v = df_v.drop(columns=drop_v)

    df_c = dataset["candidates"].copy()
    drop_c = [c for c in df_c.columns
              if c.startswith("answer_") and c not in keep_answer]
    df_c = df_c.drop(columns=drop_c)

    df_v = SVDataFrame(df_v, term=2023)
    df_c = SVDataFrame(df_c, term=2023)

    return {"questions": df_q, "voters": df_v, "candidates": df_c}


def compute_crw_recs(config, dataset, calculator):
    """Compute distances, CRW weights, and CRW recommendations for a dataset."""
    dist_df = calculator.calculate_distance(dataset, config)
    reweighter = CloneRobustReweighter(config)
    weights_df = reweighter.reweight(dist_df)

    # Build CRW recommendations manually (bypass cache for mini)
    n_cands = len(dataset["candidates"])
    voters = dataset["voters"].copy()

    # Set all weights to 1 first
    weight_cols = voters.filter(like="weight_").columns
    voters[weight_cols] = 1

    # Inject CRW weights
    weight_lookup = weights_df.set_index("ID_question")["Weight"].to_dict()
    for q_id, crw_val in weight_lookup.items():
        col = f"weight_{q_id}"
        if col in voters.columns:
            voters[col] *= crw_val

    crw_recs = add_candidate_voting_recommendations(
        df_voters=voters,
        df_candidates=dataset["candidates"],
        distance_method=config.rec_dist_method,
        n_recommendations=n_cands,
    )
    return crw_recs, weights_df


def main():
    # 1. Load config and override alpha
    config = load_config(CONFIG_PATH)
    config.alpha = ALPHA
    print(f"Config: {config.dist}, alpha={config.alpha}, district={config.district}")

    # 2. Load full dataset
    print("\nLoading dataset...")
    dataset = load_dataset(config)

    # 3. Identify mini questions
    df_questions = dataset["questions"]
    mini_mask = df_questions["rapide"] == 1
    mini_ids = sorted(df_questions.loc[mini_mask, "ID_question"].tolist())
    full_only_ids = sorted(df_questions.loc[~mini_mask, "ID_question"].tolist())
    print(f"Mini questions: {len(mini_ids)}, Full-only: {len(full_only_ids)}")

    # 4. Full questionnaire: CRW recommendations (cached pipeline)
    print("\n=== Full questionnaire: distances + CRW weights + recommendations ===")
    calculator = get_calculator(config)
    dist_df_full = calculator.calculate_distance(dataset, config)
    reweighter_full = CloneRobustReweighter(config)
    weights_df_full = reweighter_full.reweight(dist_df_full)
    rec_engine = RecommendationEngine(config=config, data_map=dataset)
    full_recs = rec_engine.evaluate_pipeline(df_weights=weights_df_full)

    # Extract CRW rankings from cached full pipeline recs
    full_rankings = CrossRunAnalyzer._extract_rankings(full_recs)

    # 5. Mini questionnaire: compute distances + CRW + recommendations
    print("\n=== Mini questionnaire: distances + CRW weights + recommendations ===")
    mini_dataset = filter_to_mini(dataset)
    mini_crw_recs, weights_df_mini = compute_crw_recs(config, mini_dataset, calculator)
    mini_rankings = CrossRunAnalyzer._extract_rankings(mini_crw_recs)

    # Also compute baseline-full for reference
    print("\n=== Extracting baseline-full rankings ===")
    # full_recs already has both baseline (_matchID_*) and CRW (CRW__matchID_*) columns

    # 6. Compare recommendations
    print("\n=== Comparing recommendations ===")
    common_voters = full_rankings.index.intersection(mini_rankings.index)
    print(f"Common voters: {len(common_voters)}")

    results = []
    for vid in common_voters:
        full_std = full_rankings.loc[vid, "ranked_standard"]
        full_crw = full_rankings.loc[vid, "ranked_crw"]
        mini_crw = mini_rankings.loc[vid, "ranked_standard"]  # CRW recs in _matchID_* cols

        # Jaccard
        jac_base = _jaccard(full_std, mini_crw, N_JACCARD)
        jac_crw = _jaccard(full_crw, mini_crw, N_JACCARD)

        # Spearman
        sp_base = _spearman(full_std, mini_crw)
        sp_crw = _spearman(full_crw, mini_crw)

        results.append({
            "voterID": vid,
            "jac_baseline_full_vs_crw_mini": jac_base,
            "jac_crw_full_vs_crw_mini": jac_crw,
            "sp_baseline_full_vs_crw_mini": sp_base,
            "sp_crw_full_vs_crw_mini": sp_crw,
        })

    df_results = pd.DataFrame(results)

    # Build output text
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    jac_b = df_results["jac_baseline_full_vs_crw_mini"]
    jac_c = df_results["jac_crw_full_vs_crw_mini"]
    sp_b = df_results["sp_baseline_full_vs_crw_mini"]
    sp_c = df_results["sp_crw_full_vs_crw_mini"]

    p(f"{'='*70}")
    p(f"RESULTS: Mini vs Full (E5-INSTRUCT, alpha={ALPHA}, ZH, top-{N_JACCARD})")
    p(f"  Comparison: CRW(full 75q) vs CRW(mini 30q)")
    p(f"  Reference:  Baseline(full 75q) vs CRW(mini 30q)")
    p(f"{'='*70}")

    p(f"\n  {'Metric':<45} {'Base-Full':>10} {'CRW-Full':>10} {'Delta':>10}")
    p(f"  {'-'*75}")
    p(f"  {'Jaccard mean (vs CRW-mini):':<45} {jac_b.mean():>10.4f} {jac_c.mean():>10.4f} {jac_c.mean()-jac_b.mean():>+10.4f}")
    p(f"  {'Jaccard median (vs CRW-mini):':<45} {jac_b.median():>10.4f} {jac_c.median():>10.4f} {jac_c.median()-jac_b.median():>+10.4f}")
    p(f"  {'Jaccard p10 (vs CRW-mini):':<45} {jac_b.quantile(0.1):>10.4f} {jac_c.quantile(0.1):>10.4f} {jac_c.quantile(0.1)-jac_b.quantile(0.1):>+10.4f}")
    p(f"  {'Spearman mean (vs CRW-mini):':<45} {sp_b.mean():>10.4f} {sp_c.mean():>10.4f} {sp_c.mean()-sp_b.mean():>+10.4f}")

    p(f"\n  % voters where CRW-full closer to CRW-mini (Jaccard): {(jac_c > jac_b).mean()*100:.1f}%")
    p(f"  % voters where baseline-full closer:                    {(jac_b > jac_c).mean()*100:.1f}%")
    p(f"  % voters unchanged:                                     {(jac_b == jac_c).mean()*100:.1f}%")

    # 7. CRW weights by topic category (full questionnaire)
    p(f"\n{'='*70}")
    p(f"CRW WEIGHTS BY TOPIC CATEGORY — FULL 75q (E5-INSTRUCT, alpha={ALPHA})")
    p(f"{'='*70}")

    q_info = df_questions[["ID_question", "_category", "rapide"]].copy()
    w_merged_full = weights_df_full.merge(q_info, on="ID_question")

    cat_summary = w_merged_full.groupby("_category").agg(
        n_questions=("Weight", "count"),
        n_mini=("rapide", lambda x: int((x == 1).sum())),
        mean_weight=("Weight", "mean"),
        std_weight=("Weight", "std"),
        min_weight=("Weight", "min"),
        max_weight=("Weight", "max"),
    ).sort_values("mean_weight")

    p(f"\n  {'Category':<38} {'N':>3} {'Mini':>4} {'Mean W':>7} {'Std':>6} {'Min W':>7} {'Max W':>7}")
    p(f"  {'-'*78}")
    for cat, row in cat_summary.iterrows():
        p(f"  {cat:<38} {int(row['n_questions']):>3} {int(row['n_mini']):>4} "
          f"{row['mean_weight']:>7.4f} {row['std_weight']:>6.3f} "
          f"{row['min_weight']:>7.4f} {row['max_weight']:>7.4f}")

    overall_w = weights_df_full["Weight"]
    p(f"\n  Overall: mean={overall_w.mean():.4f}, std={overall_w.std():.4f}, "
      f"min={overall_w.min():.4f}, max={overall_w.max():.4f}")

    # Mini CRW weights
    p(f"\n{'='*70}")
    p(f"CRW WEIGHTS BY TOPIC CATEGORY — MINI 30q (E5-INSTRUCT, alpha={ALPHA})")
    p(f"{'='*70}")

    q_info_mini = q_info[q_info["ID_question"].isin(mini_ids)]
    w_merged_mini = weights_df_mini.merge(q_info_mini, on="ID_question")

    cat_summary_mini = w_merged_mini.groupby("_category").agg(
        n_questions=("Weight", "count"),
        mean_weight=("Weight", "mean"),
        std_weight=("Weight", "std"),
        min_weight=("Weight", "min"),
        max_weight=("Weight", "max"),
    ).sort_values("mean_weight")

    p(f"\n  {'Category':<38} {'N':>3} {'Mean W':>7} {'Std':>6} {'Min W':>7} {'Max W':>7}")
    p(f"  {'-'*72}")
    for cat, row in cat_summary_mini.iterrows():
        p(f"  {cat:<38} {int(row['n_questions']):>3} "
          f"{row['mean_weight']:>7.4f} {row['std_weight']:>6.3f} "
          f"{row['min_weight']:>7.4f} {row['max_weight']:>7.4f}")

    overall_mini_w = weights_df_mini["Weight"]
    p(f"\n  Overall: mean={overall_mini_w.mean():.4f}, std={overall_mini_w.std():.4f}, "
      f"min={overall_mini_w.min():.4f}, max={overall_mini_w.max():.4f}")

    # Per-question detail (full)
    p(f"\n  --- Per-question CRW weights, full 75q (sorted by weight) ---")
    w_sorted = w_merged_full.sort_values("Weight")
    for _, row in w_sorted.iterrows():
        rapide_str = "MINI" if row["rapide"] == 1 else "full"
        text = str(row.get("Question", ""))[:55]
        p(f"    Q{int(row['ID_question']):>5}  w={row['Weight']:.4f}  [{rapide_str:>4}]  "
          f"{row['_category']:<35} {text}")

    # Verdict
    delta_jac = jac_c.mean() - jac_b.mean()
    p(f"\n{'='*70}")
    p(f"VERDICT")
    p(f"{'='*70}")
    if delta_jac > 0.005:
        p(f"  Signal detected: CRW-full is closer to CRW-mini (delta={delta_jac:+.4f}).")
        p(f"  Worth running the full Experiment 3 sweep.")
    elif delta_jac < -0.005:
        p(f"  CRW moves AWAY from CRW-mini (delta={delta_jac:+.4f}).")
        p(f"  CRW is not rebalancing toward mini. Skip Experiment 3.")
    else:
        p(f"  No meaningful difference (delta={delta_jac:+.4f}).")
        p(f"  CRW does not meaningfully rebalance natural topic imbalance.")
        p(f"  Useful finding for thesis limitations section.")

    # 8. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Save per-voter CSV
    csv_path = OUTPUT_DIR / f"exp3_sanity_check_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)

    # Save CRW weights CSVs
    weights_full_path = OUTPUT_DIR / f"exp3_crw_weights_full_{timestamp}.csv"
    w_merged_full.to_csv(weights_full_path, index=False)

    weights_mini_path = OUTPUT_DIR / f"exp3_crw_weights_mini_{timestamp}.csv"
    w_merged_mini.to_csv(weights_mini_path, index=False)

    # Save text report
    report_path = OUTPUT_DIR / f"exp3_sanity_check_{timestamp}.txt"
    report_path.write_text("\n".join(lines))

    print(f"\n--- Saved to {OUTPUT_DIR}/ ---")
    print(f"  {csv_path.name}")
    print(f"  {weights_full_path.name}")
    print(f"  {weights_mini_path.name}")
    print(f"  {report_path.name}")


def _jaccard(list_a: list, list_b: list, n: int) -> float:
    s1, s2 = set(list_a[:n]), set(list_b[:n])
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def _spearman(list_a: list, list_b: list) -> float:
    common = set(list_a) & set(list_b)
    if len(common) < 2:
        return np.nan
    rank_b = {c: i for i, c in enumerate(list_b)}
    ranks_b = np.array([rank_b[c] for c in list_a if c in rank_b])
    ranks_a = np.arange(len(ranks_b))
    s, _ = spearmanr(ranks_a, ranks_b)
    return s


if __name__ == "__main__":
    main()
