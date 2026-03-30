"""
Mini vs Maxi Distance Map

Produces 2D scatter plots and pairwise distance CSVs for all 75 questions,
color-coded by questionnaire membership:
  - mini      : in the rapide (short) questionnaire (30 questions)
  - corr_maxi : top-k most correlated full-only questions (Exp 3 "approximate clones")
  - maxi      : remaining full-only questions

Two metrics: ANSWER-CORRELATION-ARCCOS and E5-INSTRUCT.

Usage:
    # Recompute corr_maxi from voter data (top-k by max |Pearson r| to any mini question)
    python -m experiments.exploratory.analysis.mini_maxi_distance_map \\
        --config configs/full_pipeline/base_data/pipeline_answer_corr_ZH.py

    # Explicit corr_maxi IDs (e.g. for party impact analysis with a custom selection)
    python -m experiments.exploratory.analysis.mini_maxi_distance_map \\
        --corr-maxi-ids 32287,32285,32281,32276,32266

    # Different top-k
    python -m experiments.exploratory.analysis.mini_maxi_distance_map \\
        --config configs/full_pipeline/base_data/pipeline_answer_corr_ZH.py --top-k 10

Outputs (saved to experiment_results/distance_analysis/mini_maxi_map/):
    - {metric}_map_{ts}.png           -- 2D MDS scatter plot
    - {metric}_distances_{ts}.csv     -- pairwise distances with group_ID1/group_ID2 columns
    - {metric}_positions_{ts}.csv     -- per-question MDS coords + group
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

QUESTIONS_PATH = Path("data/cleaned/df_questions.parquet")

# Same cached distance files as in distance_correlation_analysis.py §3
_METRICS = {
    "ANSWER-CORRELATION-ARCCOS": Path(
        "cache/distance_calculations/dist_2023_ANSWER-CORRELATION-ARCCOS_790a61671ea5.parquet"
    ),
    "E5-INSTRUCT": Path(
        "cache/distance_calculations/dist_2023_E5-INSTRUCT_ba053f9f59a3.parquet"
    ),
}

OUTPUT_DIR = Path("experiment_results/distance_analysis/mini_maxi_map")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="2D MDS distance map: mini vs corr_maxi vs maxi questions"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Pipeline config path; loads voter data to compute corr_maxi by max |Pearson r|",
    )
    group.add_argument(
        "--corr-maxi-ids",
        type=str,
        help="Comma-separated question IDs for the corr_maxi group (e.g. 32287,32285,...)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of corr_maxi questions to select when using --config (default: 5)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_distance_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Similarity" in df.columns and "Distance" not in df.columns:
        df["Distance"] = np.sqrt(np.maximum(0, 2 * (1 - df["Similarity"])))
    # Original questions only (clones have ID >= 9_000_000)
    df = df[(df["ID1"] < 9_000_000) & (df["ID2"] < 9_000_000)].copy()
    return df


def _build_distance_matrix(dist_df: pd.DataFrame, question_ids: list) -> np.ndarray:
    id_to_idx = {qid: i for i, qid in enumerate(question_ids)}
    n = len(question_ids)
    mat = np.zeros((n, n))
    for _, row in dist_df.iterrows():
        i = id_to_idx.get(int(row["ID1"]))
        j = id_to_idx.get(int(row["ID2"]))
        if i is not None and j is not None:
            mat[i, j] = row["Distance"]
            mat[j, i] = row["Distance"]
    return mat


def _assign_group(q_id: int, mini_ids: set, corr_maxi_ids: set) -> str:
    if q_id in mini_ids:
        return "mini"
    if q_id in corr_maxi_ids:
        return "corr_maxi"
    return "maxi"


# ---------------------------------------------------------------------------
# Corr-maxi selection from voter data
# ---------------------------------------------------------------------------


def _compute_corr_maxi_from_config(
    config_path: str, mini_ids: set, full_only_ids: list, top_k: int
) -> list:
    from vqs.config_utils import load_config
    from vqs.data_loader import load_dataset
    from experiments.natural_redundancy.mini_maxi_party_impact import compute_redundancy_scores

    config = load_config(config_path)
    dataset = load_dataset(config)
    voters_df = dataset["voters"]

    redundancy = compute_redundancy_scores(voters_df, mini_ids, full_only_ids)
    ranked = sorted(redundancy.items(), key=lambda x: x[1]["max_abs_r"], reverse=True)
    top = [q_id for q_id, _ in ranked[:top_k]]
    print(f"  Top-{top_k} corr_maxi questions by max |Pearson r| to any mini question:")
    for q_id, stats in ranked[:top_k]:
        print(f"    Q{q_id}  max|r|={stats['max_abs_r']:.4f}  mean|r|={stats['mean_abs_r']:.4f}")
    return top


# ---------------------------------------------------------------------------
# Per-metric processing
# ---------------------------------------------------------------------------


def _plot_map(
    positions_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    stress: float,
):
    fig, ax = plt.subplots(figsize=(12, 10))

    group_cfg = {
        "mini": {
            "color": "#E53935",
            "label": "Mini (rapide)",
            "marker": "o",
            "s": 80,
            "zorder": 3,
        },
        "corr_maxi": {
            "color": "#43A047",
            "label": f"corr_maxi (top-k added)",
            "marker": "D",
            "s": 120,
            "zorder": 4,
        },
        "maxi": {
            "color": "#1E88E5",
            "label": "Full-only (maxi)",
            "marker": "o",
            "s": 50,
            "zorder": 2,
        },
    }

    for group, cfg in group_cfg.items():
        mask = positions_df["group"] == group
        if not mask.any():
            continue
        sub = positions_df[mask]
        ax.scatter(
            sub["mds_x"],
            sub["mds_y"],
            c=cfg["color"],
            label=f"{cfg['label']} (n={mask.sum()})",
            marker=cfg["marker"],
            s=cfg["s"],
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            zorder=cfg["zorder"],
        )
        if group == "corr_maxi":
            for _, row in sub.iterrows():
                ax.annotate(
                    f"Q{int(row['question_id'])}",
                    (row["mds_x"], row["mds_y"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                    color=cfg["color"],
                )

    ax.set_title(
        f"Question Map — {metric}\n"
        f"MDS 2D projection of pairwise distances  (stress={stress:.2f})",
        fontsize=13,
    )
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _run_metric(
    metric: str,
    dist_path: Path,
    question_ids: list,
    questions_df: pd.DataFrame,
    mini_ids: set,
    corr_maxi_ids: set,
    output_dir: Path,
    ts: str,
):
    print(f"\n--- {metric} ---")

    dist_df = _load_distance_parquet(dist_path)

    # Restrict to questions present in the distance file
    present = set(dist_df["ID1"].tolist()) | set(dist_df["ID2"].tolist())
    q_ids = [qid for qid in question_ids if qid in present]
    print(f"  Questions in distance file: {len(q_ids)}")

    mat = _build_distance_matrix(dist_df, q_ids)

    print("  Running MDS...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
    coords = mds.fit_transform(mat)
    stress = mds.stress_
    print(f"  MDS stress: {stress:.2f}")

    # Question metadata
    text_col = next(
        (c for c in ["question_EN", "question_en"] if c in questions_df.columns), None
    )
    id_to_text = {
        int(row["ID_question"]): str(row[text_col]) if text_col else str(row["ID_question"])
        for _, row in questions_df.iterrows()
    }
    id_to_cat = {
        int(row["ID_question"]): row.get("_category", "")
        for _, row in questions_df.iterrows()
    }

    # Positions CSV
    positions_rows = [
        {
            "question_id": qid,
            "question_text": id_to_text.get(qid, ""),
            "category": id_to_cat.get(qid, ""),
            "group": _assign_group(qid, mini_ids, corr_maxi_ids),
            "mds_x": coords[idx, 0],
            "mds_y": coords[idx, 1],
        }
        for idx, qid in enumerate(q_ids)
    ]
    positions_df = pd.DataFrame(positions_rows)

    metric_slug = metric.lower().replace("-", "_")
    pos_path = output_dir / f"{metric_slug}_positions_{ts}.csv"
    positions_df.to_csv(pos_path, index=False)
    print(f"  Saved: {pos_path}  ({len(positions_df)} rows)")

    # Distances CSV with group columns
    id_to_group = dict(zip(positions_df["question_id"], positions_df["group"]))
    dist_df["group_ID1"] = dist_df["ID1"].apply(lambda x: id_to_group.get(int(x), "unknown"))
    dist_df["group_ID2"] = dist_df["ID2"].apply(lambda x: id_to_group.get(int(x), "unknown"))
    dist_out = output_dir / f"{metric_slug}_distances_{ts}.csv"
    dist_df[["ID1", "ID2", "Qu1", "Qu2", "Distance", "group_ID1", "group_ID2"]].to_csv(
        dist_out, index=False
    )
    print(f"  Saved: {dist_out}  ({len(dist_df)} rows)")

    # Scatter plot
    map_path = output_dir / f"{metric_slug}_map_{ts}.png"
    _plot_map(positions_df, metric, map_path, stress)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    ts = datetime.now().strftime("%m%d_%H%M")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load questions metadata
    questions_df = pd.read_parquet(QUESTIONS_PATH)
    all_ids = [int(x) for x in questions_df["ID_question"].tolist()]
    mini_ids = set(
        int(x) for x in questions_df.loc[questions_df["rapide"] == 1, "ID_question"]
    )
    full_only_ids = [qid for qid in all_ids if qid not in mini_ids]

    print(f"Mini questions:      {len(mini_ids)}")
    print(f"Full-only questions: {len(full_only_ids)}")

    # Determine corr_maxi group
    if args.corr_maxi_ids:
        corr_maxi_ids = set(int(x.strip()) for x in args.corr_maxi_ids.split(","))
        print(f"Using explicit corr_maxi IDs ({len(corr_maxi_ids)}): {sorted(corr_maxi_ids)}")
    else:
        corr_maxi_list = _compute_corr_maxi_from_config(
            args.config, mini_ids, full_only_ids, args.top_k
        )
        corr_maxi_ids = set(corr_maxi_list)

    maxi_count = len(full_only_ids) - len(corr_maxi_ids)
    print(f"corr_maxi:           {len(corr_maxi_ids)}")
    print(f"maxi (rest):         {maxi_count}")

    for metric, dist_path in _METRICS.items():
        _run_metric(
            metric=metric,
            dist_path=dist_path,
            question_ids=all_ids,
            questions_df=questions_df,
            mini_ids=mini_ids,
            corr_maxi_ids=corr_maxi_ids,
            output_dir=OUTPUT_DIR,
            ts=ts,
        )

    print(f"\nDone. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
