"""
Analyze pairwise question distance matrices.

Reads a distance parquet from experiment_results/pipeline_outputs/distance_metrics/ (or cache/),
produces summary statistics and a visualization of the distance space.

Outputs saved to experiment_results/distance_analysis/.

Usage:
    # Analyze a specific distance file:
    python -m scripts.analyze_distances experiment_results/pipeline_outputs/distance_metrics/cleaned_results/ANSWER-CORRELATION_2023_ZH_*.parquet

    # With question metadata (adds category labels):
    python -m scripts.analyze_distances <path> --config configs/base_pipeline/pipeline_answer_corr_ZH.py

    # Top-N closest pairs (default 20):
    python -m scripts.analyze_distances <path> --top 30
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("experiment_results/distance_analysis")


def load_distances(path: str) -> pd.DataFrame:
    """Load distance parquet and normalize to a 'distance' column."""
    df = pd.read_parquet(path)

    # Normalize column name: "Similarity" → convert to distance
    if "Similarity" in df.columns and "Distance" not in df.columns:
        df["Distance"] = np.sqrt(np.maximum(0, 2 * (1 - df["Similarity"])))
    elif "Distance" not in df.columns:
        raise ValueError(f"No 'Distance' or 'Similarity' column in {path}")

    return df


def classify_pair(id1: int, id2: int) -> str:
    """Classify a question pair as original-original, clone-source, or clone-clone."""
    is_clone_1 = id1 > 9_000_000
    is_clone_2 = id2 > 9_000_000

    if not is_clone_1 and not is_clone_2:
        return "original-original"

    if is_clone_1 and is_clone_2:
        return "clone-clone"

    # One is clone, one is original — check if it's the source
    clone_id = id1 if is_clone_1 else id2
    orig_id = id2 if is_clone_1 else id1
    source_id = (clone_id - 9_000_000) // 1000
    if source_id == orig_id:
        return "clone-source"
    else:
        return "clone-other"


def analyze(df: pd.DataFrame, question_info: pd.DataFrame | None = None) -> dict:
    """Compute summary statistics from distance DataFrame."""
    distances = df["Distance"].values
    has_clones = (df["ID1"] > 9_000_000).any() or (df["ID2"] > 9_000_000).any()

    stats = {
        "n_pairs": len(df),
        "n_questions": len(set(df["ID1"]) | set(df["ID2"])),
        "mean": distances.mean(),
        "median": np.median(distances),
        "std": distances.std(),
        "min": distances.min(),
        "max": distances.max(),
        "p5": np.percentile(distances, 5),
        "p10": np.percentile(distances, 10),
        "p25": np.percentile(distances, 25),
        "p75": np.percentile(distances, 75),
        "p90": np.percentile(distances, 90),
        "p95": np.percentile(distances, 95),
        "has_clones": has_clones,
    }

    if has_clones:
        df = df.copy()
        df["pair_type"] = df.apply(lambda r: classify_pair(r["ID1"], r["ID2"]), axis=1)
        for ptype in ["original-original", "clone-source", "clone-other", "clone-clone"]:
            subset = df.loc[df["pair_type"] == ptype, "Distance"]
            if len(subset) > 0:
                stats[f"{ptype}_n"] = len(subset)
                stats[f"{ptype}_mean"] = subset.mean()
                stats[f"{ptype}_min"] = subset.min()
                stats[f"{ptype}_max"] = subset.max()

    return stats


def get_top_pairs(df: pd.DataFrame, n: int, question_info: pd.DataFrame | None = None) -> pd.DataFrame:
    """Get the N closest and N most distant pairs."""
    df_sorted = df.sort_values("Distance")

    closest = df_sorted.head(n).copy()
    furthest = df_sorted.tail(n).copy()

    closest["rank_type"] = "closest"
    furthest["rank_type"] = "furthest"

    result = pd.concat([closest, furthest], ignore_index=True)

    # Add pair classification
    has_clones = (df["ID1"] > 9_000_000).any() or (df["ID2"] > 9_000_000).any()
    if has_clones:
        result["pair_type"] = result.apply(lambda r: classify_pair(r["ID1"], r["ID2"]), axis=1)

    # Add category info if available
    if question_info is not None and "_category" in question_info.columns:
        cat_map = question_info.set_index("ID_question")["_category"].to_dict()
        result["cat1"] = result["ID1"].map(cat_map).fillna("clone")
        result["cat2"] = result["ID2"].map(cat_map).fillna("clone")

    return result


def plot_distance_distribution(df: pd.DataFrame, title: str, output_path: Path):
    """Plot histogram + KDE of distances, split by pair type if clones present."""
    has_clones = (df["ID1"] > 9_000_000).any() or (df["ID2"] > 9_000_000).any()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall histogram
    ax = axes[0]
    ax.hist(df["Distance"], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.set_title("All pairs")
    ax.axvline(df["Distance"].median(), color="red", linestyle="--", label=f'median={df["Distance"].median():.3f}')
    ax.axvline(df["Distance"].mean(), color="orange", linestyle="--", label=f'mean={df["Distance"].mean():.3f}')
    ax.legend(fontsize=8)

    # Right: split by pair type if clones exist, otherwise zoom on low distances
    ax = axes[1]
    if has_clones:
        df_plot = df.copy()
        df_plot["pair_type"] = df_plot.apply(lambda r: classify_pair(r["ID1"], r["ID2"]), axis=1)
        colors = {
            "original-original": "steelblue",
            "clone-source": "red",
            "clone-other": "orange",
            "clone-clone": "green",
        }
        for ptype, color in colors.items():
            subset = df_plot.loc[df_plot["pair_type"] == ptype, "Distance"]
            if len(subset) > 0:
                ax.hist(subset, bins=30, alpha=0.5, color=color, label=f"{ptype} (n={len(subset)})", edgecolor="white")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.set_title("By pair type")
        ax.legend(fontsize=7)
    else:
        # Zoom into bottom 10% of distances
        threshold = np.percentile(df["Distance"], 10)
        low = df[df["Distance"] <= threshold]
        ax.hist(low["Distance"], bins=30, alpha=0.7, color="coral", edgecolor="white")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.set_title(f"Lowest 10% (distance ≤ {threshold:.3f})")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze pairwise question distances")
    parser.add_argument("path", help="Path to distance parquet file")
    parser.add_argument("--config", help="Pipeline config path (for question metadata)", default=None)
    parser.add_argument("--top", type=int, default=20, help="Number of closest/furthest pairs to show")
    parser.add_argument("--name", help="Override subfolder name (use when filename doesn't match expected pattern)", default=None)
    args = parser.parse_args()

    # Load distances
    print(f"Loading {args.path}...")
    df = load_distances(args.path)
    print(f"  {len(df)} pairs, {len(set(df['ID1']) | set(df['ID2']))} questions")

    # Load question metadata if config provided
    question_info = None
    if args.config:
        from vqs.config_utils import load_config
        from vqs.data_loader import load_dataset
        config = load_config(Path(args.config))
        dataset = load_dataset(config)
        question_info = dataset["questions"]

    # Analyze
    stats = analyze(df, question_info)

    # Print summary
    label = Path(args.path).stem
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"{'=' * 70}")
    p(f"DISTANCE ANALYSIS: {label}")
    p(f"{'=' * 70}")
    p(f"  Questions:  {stats['n_questions']}")
    p(f"  Pairs:      {stats['n_pairs']}")
    p(f"  Has clones: {stats['has_clones']}")
    p()
    p(f"  {'Statistic':<20} {'Value':>10}")
    p(f"  {'-' * 32}")
    for key in ["mean", "median", "std", "min", "max", "p5", "p10", "p25", "p75", "p90", "p95"]:
        p(f"  {key:<20} {stats[key]:>10.4f}")

    if stats["has_clones"]:
        p()
        p(f"  {'Pair Type':<25} {'N':>6} {'Mean':>8} {'Min':>8} {'Max':>8}")
        p(f"  {'-' * 57}")
        for ptype in ["original-original", "clone-source", "clone-other", "clone-clone"]:
            n_key = f"{ptype}_n"
            if n_key in stats:
                p(f"  {ptype:<25} {stats[n_key]:>6} {stats[f'{ptype}_mean']:>8.4f} "
                  f"{stats[f'{ptype}_min']:>8.4f} {stats[f'{ptype}_max']:>8.4f}")

    # Top pairs
    p()
    p(f"{'=' * 70}")
    p(f"TOP {args.top} CLOSEST PAIRS")
    p(f"{'=' * 70}")
    top_df = get_top_pairs(df, args.top, question_info)
    closest = top_df[top_df["rank_type"] == "closest"]

    for _, row in closest.iterrows():
        id1, id2 = int(row["ID1"]), int(row["ID2"])
        dist = row["Distance"]
        q1 = str(row.get("Qu1", ""))[:45]
        q2 = str(row.get("Qu2", ""))[:45]
        extra = ""
        if "pair_type" in row:
            extra += f"  [{row['pair_type']}]"
        if "cat1" in row:
            extra += f"  {row['cat1']} / {row['cat2']}"
        p(f"  {id1:>8} — {id2:>8}  d={dist:.4f}{extra}")
        p(f"    Q1: {q1}")
        p(f"    Q2: {q2}")

    p()
    p(f"{'=' * 70}")
    p(f"TOP {args.top} MOST DISTANT PAIRS")
    p(f"{'=' * 70}")
    furthest = top_df[top_df["rank_type"] == "furthest"].sort_values("Distance", ascending=False)
    for _, row in furthest.iterrows():
        id1, id2 = int(row["ID1"]), int(row["ID2"])
        dist = row["Distance"]
        q1 = str(row.get("Qu1", ""))[:45]
        q2 = str(row.get("Qu2", ""))[:45]
        p(f"  {id1:>8} — {id2:>8}  d={dist:.4f}")
        p(f"    Q1: {q1}")
        p(f"    Q2: {q2}")

    # Build subfolder name: {METRIC}_{YEAR}[_{district}][_{clone_dataset}]
    import re
    input_path = Path(args.path)

    if args.name:
        subfolder_name = args.name
    else:
        # Filename pattern: {METRIC}_{YEAR}[_{ZH}]_{MMDD}_{HHMM}_{hash}.{ext}
        # For cloned results, parent folder is the clone dataset name.
        match = re.match(r"^([A-Z0-9-]+)_(\d{4})(?:_([A-Z]{2}))?_\d{4}_\d{4}_", label)
        if not match:
            raise ValueError(
                f"Filename '{label}' doesn't match expected pattern "
                f"'{{METRIC}}_{{YEAR}}_{{MMDD}}_{{HHMM}}_{{hash}}'. "
                f"Use --name to set the subfolder name manually."
            )
        metric, year, district_str = match.group(1), match.group(2), match.group(3)
        subfolder_parts = [metric, year]
        if district_str:
            subfolder_parts.append(district_str)

        # Check if this is a cloned dataset (parent folder is clone name, not cleaned_results)
        parent_name = input_path.parent.name
        if parent_name not in ("cleaned_results", "fake_results"):
            subfolder_parts.append(parent_name)

        subfolder_name = "_".join(subfolder_parts)

    # Save outputs
    out_dir = OUTPUT_DIR / subfolder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base_name = f"dist_analysis_{timestamp}"

    # Save stats CSV
    stats_df = pd.DataFrame([stats])
    stats_path = out_dir / f"{base_name}_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    # Save top pairs CSV
    top_path = out_dir / f"{base_name}_top_pairs.csv"
    top_df.to_csv(top_path, index=False)

    # Save full sorted distances CSV (for re-plotting)
    full_path = out_dir / f"{base_name}_all_distances.csv"
    df_out = df[["ID1", "ID2", "Distance"]].sort_values("Distance")
    if question_info is not None and "_category" in question_info.columns:
        cat_map = question_info.set_index("ID_question")["_category"].to_dict()
        df_out = df_out.copy()
        df_out["cat1"] = df_out["ID1"].map(cat_map).fillna("clone")
        df_out["cat2"] = df_out["ID2"].map(cat_map).fillna("clone")
    df_out.to_csv(full_path, index=False)

    # Save plot
    plot_path = out_dir / f"{base_name}_distribution.png"
    plot_distance_distribution(df, subfolder_name, plot_path)

    # Save report
    report_path = out_dir / f"{base_name}_report.txt"
    report_path.write_text("\n".join(lines))

    p()
    p(f"--- Saved to {out_dir}/ ---")
    p(f"  {stats_path.name}")
    p(f"  {top_path.name}")
    p(f"  {full_path.name}")
    p(f"  {plot_path.name}")
    p(f"  {report_path.name}")


if __name__ == "__main__":
    main()
