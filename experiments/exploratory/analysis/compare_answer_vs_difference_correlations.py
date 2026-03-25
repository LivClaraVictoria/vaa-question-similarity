"""
Compare answer-space vs difference-space correlations between questions.

Answer-space: correlate voter answer columns directly (75×75 matrix).
Difference-space: for sampled voter-candidate pairs, compute per-question
differences δᵢ = answer_voter(qᵢ) - answer_candidate(qᵢ), then correlate
those δ columns (75×75 matrix).

The hypothesis: answer-space correlations overestimate true recommendation
redundancy because L2 matching operates on differences, not raw answers.

Usage:
    python -m scripts.compare_answer_vs_difference_correlations
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import seaborn as sns

from vqs.config_utils import load_config
from vqs.data_loader import load_dataset

CONFIG_PATH = "configs/full_pipeline/base_data/pipeline_e5_ZH.py"
OUTPUT_DIR = Path("experiment_results/question_impact/difference_space_analysis")
N_SAMPLE_VOTERS = 1000
SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(CONFIG_PATH))
    dataset = load_dataset(config)
    df_voters = dataset["voters"]
    df_candidates = dataset["candidates"]

    # Get answer columns present in both
    question_ids = sorted(
        int(c.replace("answer_", ""))
        for c in df_voters.columns
        if c.startswith("answer_") and c in df_candidates.columns
    )
    ans_cols = [f"answer_{q}" for q in question_ids]
    print(f"Questions: {len(question_ids)}, Voters: {len(df_voters)}, Candidates: {len(df_candidates)}")

    # --- Answer-space correlation (voter answers only) ---
    voter_corr = df_voters[ans_cols].corr()
    cand_corr = df_candidates[ans_cols].corr()

    # --- Difference-space correlation (sampled voter-candidate pairs) ---
    rng = np.random.default_rng(SEED)
    n_voters = min(N_SAMPLE_VOTERS, len(df_voters))
    voter_idx = rng.choice(len(df_voters), size=n_voters, replace=False)
    sampled_voters = df_voters.iloc[voter_idx][ans_cols].values  # (n_voters, 75)
    all_candidates = df_candidates[ans_cols].values  # (n_cands, 75)

    # Compute all differences: (n_voters * n_cands, 75)
    print(f"Computing differences for {n_voters} voters × {len(all_candidates)} candidates = {n_voters * len(all_candidates):,} pairs...")
    diffs = sampled_voters[:, np.newaxis, :] - all_candidates[np.newaxis, :, :]  # (n_voters, n_cands, 75)
    diffs = diffs.reshape(-1, len(question_ids))  # (n_pairs, 75)

    # Drop rows with any NaN
    mask = ~np.isnan(diffs).any(axis=1)
    diffs_clean = diffs[mask]
    print(f"  Clean pairs (no NaN): {len(diffs_clean):,} / {len(diffs):,}")

    diff_corr = np.corrcoef(diffs_clean.T)  # (75, 75)
    diff_corr_df = pd.DataFrame(diff_corr, index=ans_cols, columns=ans_cols)

    # --- Extract upper triangle (all unique pairs) ---
    n_q = len(question_ids)
    triu_idx = np.triu_indices(n_q, k=1)

    voter_abs_r = np.abs(voter_corr.values[triu_idx])
    cand_abs_r = np.abs(cand_corr.values[triu_idx])
    diff_abs_r = np.abs(diff_corr[triu_idx])

    print(f"\n--- Summary ({len(voter_abs_r)} question pairs) ---")
    for name, vals in [("Voter answer", voter_abs_r), ("Candidate answer", cand_abs_r), ("Difference", diff_abs_r)]:
        print(f"  {name:20s}:  mean |r| = {vals.mean():.4f},  max |r| = {vals.max():.4f},  median = {np.median(vals):.4f}")

    # Spearman between the two orderings
    rho_vd, p_vd = spearmanr(voter_abs_r, diff_abs_r)
    rho_cd, p_cd = spearmanr(cand_abs_r, diff_abs_r)
    print(f"\n  Spearman(voter answer |r|, difference |r|):    ρ = {rho_vd:.4f}, p = {p_vd:.2e}")
    print(f"  Spearman(candidate answer |r|, difference |r|): ρ = {rho_cd:.4f}, p = {p_cd:.2e}")

    # --- Plot 1: Scatter comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (ans_r, label, rho, p) in zip(axes, [
        (voter_abs_r, "Voter Answer |r|", rho_vd, p_vd),
        (cand_abs_r, "Candidate Answer |r|", rho_cd, p_cd),
    ]):
        ax.scatter(ans_r, diff_abs_r, alpha=0.3, s=15, color="#1565C0")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y = x")
        ax.set_xlabel(f"{label}")
        ax.set_ylabel("Difference-Space |r|")
        ax.set_title(f"{label} vs Difference-Space |r|\n(Spearman ρ={rho:.3f}, p={p:.2e})")
        ax.set_xlim(0, max(ans_r.max(), diff_abs_r.max()) + 0.02)
        ax.set_ylim(0, max(ans_r.max(), diff_abs_r.max()) + 0.02)
        ax.set_aspect("equal")
        ax.legend()

    fig.suptitle(
        f"Answer-Space vs Difference-Space Correlations\n"
        f"({len(diffs_clean):,} voter-candidate pairs, {len(question_ids)} questions)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "answer_vs_difference_scatter.png", dpi=300)
    plt.close(fig)
    print(f"\n  -> Scatter: {OUTPUT_DIR / 'answer_vs_difference_scatter.png'}")

    # --- Plot 2: Side-by-side heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    q_labels = [str(q) for q in question_ids]

    for ax, mat, title in zip(axes, [voter_corr.values, cand_corr.values, diff_corr], [
        "Voter Answer Correlation", "Candidate Answer Correlation", "Difference-Space Correlation"
    ]):
        sns.heatmap(
            np.abs(mat), ax=ax, vmin=0, vmax=0.8, cmap="YlOrRd",
            xticklabels=q_labels, yticklabels=q_labels,
            square=True, cbar_kws={"shrink": 0.7},
        )
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=5)

    fig.suptitle("Absolute Correlation Matrices: Answer-Space vs Difference-Space", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmaps.png", dpi=300)
    plt.close(fig)
    print(f"  -> Heatmaps: {OUTPUT_DIR / 'correlation_heatmaps.png'}")

    # --- Plot 3: Distribution comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 0.8, 40)
    ax.hist(voter_abs_r, bins=bins, alpha=0.5, label=f"Voter answer (mean={voter_abs_r.mean():.3f})", color="#1565C0")
    ax.hist(cand_abs_r, bins=bins, alpha=0.5, label=f"Candidate answer (mean={cand_abs_r.mean():.3f})", color="#43A047")
    ax.hist(diff_abs_r, bins=bins, alpha=0.5, label=f"Difference-space (mean={diff_abs_r.mean():.3f})", color="#E53935")
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("Count (question pairs)")
    ax.set_title("Distribution of Pairwise |r| Across Correlation Spaces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_distributions.png", dpi=300)
    plt.close(fig)
    print(f"  -> Distributions: {OUTPUT_DIR / 'correlation_distributions.png'}")

    # --- Save CSV ---
    pairs = []
    for i, j in zip(*triu_idx):
        pairs.append({
            "question_a": question_ids[i],
            "question_b": question_ids[j],
            "voter_abs_r": voter_abs_r[len(pairs)],
            "candidate_abs_r": cand_abs_r[len(pairs)],
            "difference_abs_r": diff_abs_r[len(pairs)],
        })
    pd.DataFrame(pairs).to_csv(OUTPUT_DIR / "pairwise_correlations.csv", index=False)
    print(f"  -> CSV: {OUTPUT_DIR / 'pairwise_correlations.csv'}")


if __name__ == "__main__":
    main()
