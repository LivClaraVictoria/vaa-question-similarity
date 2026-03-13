# -*- coding: utf-8 -*-
"""
Compile Experiment 1 alpha sweep results: 3 clone conditions x 10 models.

Reads all Gen2 (top5impact_n4) alpha sweep CSVs and produces:
  - Per-metric comparison plots (9): all models on one axes per (clone, metric)
  - Model ranking tables (4 CSVs): per-clone + aggregated
  - Model ranking bar charts (4): grouped bars per model
  - Narrative report (1 txt): peaked vs stable analysis + benchmark correlation

Usage:
    python scripts/compile_exp1_alpha_sweep.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats
import re
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("experiment_results/exp1/model_alpha_sweep/top5impact")
BENCHMARK_CSV = Path("experiment_results/model_benchmark/model_comparison.csv")
OUTPUT_DIR = Path("experiment_results/exp1/model_alpha_sweep/compiled")

# ---------------------------------------------------------------------------
# Gen2 filter
# ---------------------------------------------------------------------------
GEN2_FILTER = "top5impact_n4"

# ---------------------------------------------------------------------------
# Models and clone conditions
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    "sbert", "e5", "e5_instruct", "e5_asym", "e5_asym_instruct",
    "jina_v3", "bge_m3", "gte", "nomic_v2", "qwen3",
]
MODEL_DISPLAY = {
    "sbert": "SBERT",
    "e5": "E5",
    "e5_instruct": "E5-Instruct",
    "e5_asym": "E5-Asym",
    "e5_asym_instruct": "E5-Asym-Instruct",
    "jina_v3": "Jina v3",
    "bge_m3": "BGE-M3",
    "gte": "GTE",
    "nomic_v2": "Nomic v2",
    "qwen3": "Qwen3",
}

CLONE_ORDER = ["easy_paraphrase", "negation_easy", "mixed"]
CLONE_DISPLAY = {
    "easy_paraphrase": "Easy Paraphrase",
    "negation_easy": "Negation + Easy Paraphrase",
    "mixed": "Mixed",
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
METRICS = {
    "jaccard": {"col": "crw_jaccard_mean", "label": "Jaccard Similarity (mean)"},
    "spearman": {"col": "crw_spearman_mean", "label": "Spearman Correlation (mean)"},
    "kendall": {"col": "crw_kendall_mean", "label": "Kendall Correlation (mean)"},
}

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
MODEL_COLORS = {m: plt.cm.tab10(i) for i, m in enumerate(MODEL_ORDER)}

# Ranking bar chart colors (matching question impact ranking style)
_BAR_JACCARD = "#E53935"
_BAR_SPEARMAN = "#1E88E5"
_BAR_KENDALL = "#43A047"

# Benchmark model name mapping (uppercase-hyphen -> our keys)
_BENCHMARK_TO_KEY = {
    "SBERT": "sbert",
    "SBERT-EUCLIDEAN": "sbert",  # same model, different distance
    "E5": "e5",
    "E5-INSTRUCT": "e5_instruct",
    "E5-ASYMMETRIC": "e5_asym",
    "E5-ASYMMETRIC-INSTRUCT": "e5_asym_instruct",
    "JINA-V3": "jina_v3",
    "BGE-M3": "bge_m3",
    "GTE": "gte",
    "NOMIC-V2": "nomic_v2",
    "QWEN3": "qwen3",
}


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_gen2_dirname(dirname: str) -> tuple[str, str] | None:
    """Extract (model, clone_type) from a Gen2 alpha sweep directory name.

    Returns None if not a Gen2 directory.
    """
    if GEN2_FILTER not in dirname:
        return None

    parts = dirname.split("_vs_", 1)
    if len(parts) != 2:
        return None

    config_a = parts[0]
    config_b = parts[1]

    # config_a: alpha_sweep_pipeline_{model}_ZH
    model = config_a.replace("alpha_sweep_pipeline_", "").removesuffix("_ZH")

    # config_b: {clone_type}_top5impact_n4_{model}_ZH
    # Remove the suffix: _top5impact_n4_{model}_ZH
    suffix = f"_top5impact_n4_{model}_ZH"
    if not config_b.endswith(suffix):
        return None
    clone_type = config_b[: -len(suffix)]

    return model, clone_type


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gen2_data() -> pd.DataFrame:
    """Load all Gen2 alpha sweep CSVs into a single DataFrame."""
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    frames = []
    found = set()

    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_gen2_dirname(subdir.name)
        if parsed is None:
            continue

        model, clone_type = parsed

        # Find aggregated CSV (not in workers_* subdirs)
        csvs = sorted(
            f for f in subdir.glob("alpha_sweep_*.csv")
            if "worker" not in f.name
        )
        if not csvs:
            continue

        # Take latest by timestamp in filename
        csv_path = csvs[-1]
        df = pd.read_csv(csv_path)
        df["model"] = model
        df["clone_type"] = clone_type

        # Extract timestamp for dedup
        match = re.search(r"_(\d{4})_(\d{4})_[a-f0-9]+\.csv$", csv_path.name)
        df["_ts"] = match.group(1) + match.group(2) if match else "0000"

        frames.append(df)
        found.add((model, clone_type))

    if not frames:
        print(f"ERROR: No Gen2 alpha sweep CSVs found in {RESULTS_DIR}")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep latest per (model, clone_type, alpha)
    combined = combined.sort_values("_ts").drop_duplicates(
        subset=["model", "clone_type", "alpha"], keep="last"
    )
    combined = combined.drop(columns=["_ts"])

    # Report coverage
    expected = {(m, c) for m in MODEL_ORDER for c in CLONE_ORDER}
    missing = expected - found
    if missing:
        print(f"  WARNING: Missing {len(missing)} sweep(s):")
        for m, c in sorted(missing):
            print(f"    - {MODEL_DISPLAY.get(m, m)} x {CLONE_DISPLAY.get(c, c)}")

    print(f"  Loaded {len(found)} model x clone combinations "
          f"({len(combined)} total rows)")
    return combined


def extract_baselines(combined: pd.DataFrame) -> dict[str, float]:
    """Extract baseline metric values from alpha=0.01 rows.

    Baselines are identical across all models (baseline doesn't use embeddings).
    """
    baseline_rows = combined[combined["alpha"] == 0.01]

    baselines = {}
    for metric_key, cfg in METRICS.items():
        vals = baseline_rows[cfg["col"]]
        baselines[metric_key] = vals.mean()
        std = vals.std()
        if std > 1e-6:
            print(f"  WARNING: Baseline {metric_key} varies across models "
                  f"(std={std:.6f})")

    return baselines


# ---------------------------------------------------------------------------
# Ranking tables
# ---------------------------------------------------------------------------

def compute_ranking_table(
    combined: pd.DataFrame, clone_type: str,
) -> pd.DataFrame:
    """Compute model ranking table for a single clone condition."""
    clone_data = combined[combined["clone_type"] == clone_type]

    rows = []
    for model in MODEL_ORDER:
        model_data = clone_data[clone_data["model"] == model]
        if model_data.empty:
            continue

        # Exclude alpha=0.01 (baseline proxy) for peak computation
        sweep_data = model_data[model_data["alpha"] > 0.02].sort_values("alpha")

        max_jaccard = sweep_data["crw_jaccard_mean"].max()
        max_spearman = sweep_data["crw_spearman_mean"].max()
        max_kendall = sweep_data["crw_kendall_mean"].max()

        # Optimal alpha (where Jaccard is maximized)
        best_idx = sweep_data["crw_jaccard_mean"].idxmax()
        optimal_alpha = sweep_data.loc[best_idx, "alpha"]

        # Alpha 95% interval: alphas where Jaccard >= 95% of max
        threshold = max_jaccard * 0.95
        in_range = sweep_data[sweep_data["crw_jaccard_mean"] >= threshold]
        alpha_95_low = in_range["alpha"].min()
        alpha_95_high = in_range["alpha"].max()

        rows.append({
            "model": MODEL_DISPLAY[model],
            "max_jaccard": max_jaccard,
            "max_spearman": max_spearman,
            "max_kendall": max_kendall,
            "optimal_alpha_jaccard": optimal_alpha,
            "alpha_95_interval": f"[{alpha_95_low:.1f}, {alpha_95_high:.1f}]",
            "alpha_95_low": alpha_95_low,
            "alpha_95_high": alpha_95_high,
            "alpha_95_width": alpha_95_high - alpha_95_low,
        })

    df = pd.DataFrame(rows)

    # Compute ranks (higher metric = rank 1)
    df["rank_jaccard"] = df["max_jaccard"].rank(ascending=False, method="min").astype(int)
    df["rank_spearman"] = df["max_spearman"].rank(ascending=False, method="min").astype(int)
    df["rank_kendall"] = df["max_kendall"].rank(ascending=False, method="min").astype(int)
    df["composite_rank"] = (
        df["rank_jaccard"] + df["rank_spearman"] + df["rank_kendall"]
    ) / 3.0

    df = df.sort_values("composite_rank").reset_index(drop=True)
    return df


def compute_aggregated_ranking(
    per_clone_rankings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Average composite_rank and max metrics across clone conditions."""
    all_models = set()
    for df in per_clone_rankings.values():
        all_models.update(df["model"].tolist())

    rows = []
    for model in sorted(all_models):
        ranks = []
        jaccards = []
        spearmans = []
        kendalls = []
        for clone, df in per_clone_rankings.items():
            model_row = df[df["model"] == model]
            if not model_row.empty:
                ranks.append(model_row["composite_rank"].values[0])
                jaccards.append(model_row["max_jaccard"].values[0])
                spearmans.append(model_row["max_spearman"].values[0])
                kendalls.append(model_row["max_kendall"].values[0])

        if ranks:
            rows.append({
                "model": model,
                "avg_max_jaccard": np.mean(jaccards),
                "avg_max_spearman": np.mean(spearmans),
                "avg_max_kendall": np.mean(kendalls),
                "avg_composite_rank": np.mean(ranks),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("avg_composite_rank").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Plot — Per-metric comparison (9 plots)
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    combined: pd.DataFrame,
    baselines: dict[str, float],
    clone_type: str,
    metric_key: str,
    outpath: Path,
):
    """All models as colored curves for one (clone_condition, metric) pair."""
    cfg = METRICS[metric_key]
    col = cfg["col"]
    label = cfg["label"]
    baseline_val = baselines[metric_key]

    clone_data = combined[combined["clone_type"] == clone_type]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    for model in MODEL_ORDER:
        model_data = clone_data[clone_data["model"] == model].sort_values("alpha")
        if model_data.empty:
            continue

        color = MODEL_COLORS[model]
        display = MODEL_DISPLAY[model]

        ax.plot(
            model_data["alpha"], model_data[col],
            color=color, linewidth=1.8, label=display,
            marker="o", markersize=3,
        )

        # Mark optimal alpha with a larger dot
        best_idx = model_data[col].idxmax()
        best_alpha = model_data.loc[best_idx, "alpha"]
        best_val = model_data.loc[best_idx, col]
        ax.plot(
            best_alpha, best_val,
            marker="*", color=color, markersize=12, zorder=5,
        )

    # Baseline horizontal line
    ax.axhline(
        baseline_val, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"Baseline: {baseline_val:.3f}",
    )

    clone_display = CLONE_DISPLAY[clone_type]
    ax.set_xlabel(r"Alpha ($\alpha$)", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(
        f"{clone_display} \u2014 {label} vs Alpha\n"
        f"(\u2605 = optimal \u03b1 per model)",
        fontsize=13,
    )
    ax.legend(
        fontsize=9, ncol=2, loc="best",
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Ranking bar chart
# ---------------------------------------------------------------------------

def plot_ranking_bars(
    ranking_df: pd.DataFrame,
    baselines: dict[str, float],
    title: str,
    outpath: Path,
    jaccard_col: str = "max_jaccard",
    spearman_col: str = "max_spearman",
    kendall_col: str = "max_kendall",
):
    """Vertical grouped bar chart: 3 metrics per model, sorted by rank."""
    sns.set_theme(style="whitegrid")

    models = ranking_df["model"].tolist()  # already sorted by composite_rank
    n = len(models)

    fig, ax = plt.subplots(figsize=(max(12, n * 1.4), 7))

    x = np.arange(n)
    width = 0.25

    ax.bar(x - width, ranking_df[jaccard_col], width,
           label="Jaccard", color=_BAR_JACCARD)
    ax.bar(x, ranking_df[spearman_col], width,
           label="Spearman", color=_BAR_SPEARMAN)
    ax.bar(x + width, ranking_df[kendall_col], width,
           label="Kendall", color=_BAR_KENDALL)

    # Baseline horizontal lines
    ax.axhline(baselines["jaccard"], color=_BAR_JACCARD, linestyle="--",
               linewidth=1, alpha=0.5)
    ax.axhline(baselines["spearman"], color=_BAR_SPEARMAN, linestyle="--",
               linewidth=1, alpha=0.5)
    ax.axhline(baselines["kendall"], color=_BAR_KENDALL, linestyle="--",
               linewidth=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Metric Value (higher = better CRW correction)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=10)

    # Y-axis: start from a reasonable minimum so differences are visible
    all_vals = pd.concat([
        ranking_df[jaccard_col],
        ranking_df[spearman_col],
        ranking_df[kendall_col],
    ])
    y_min = min(all_vals.min(), min(baselines.values())) * 0.95
    ax.set_ylim(bottom=y_min)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_benchmark_scores() -> pd.DataFrame | None:
    """Load fake benchmark ordering accuracy scores."""
    if not BENCHMARK_CSV.is_file():
        print(f"  Benchmark CSV not found: {BENCHMARK_CSV}")
        return None

    bench = pd.read_csv(BENCHMARK_CSV)

    # Map benchmark model names to our keys
    rows = []
    for _, row in bench.iterrows():
        key = _BENCHMARK_TO_KEY.get(row["model"])
        if key and key in MODEL_ORDER:
            rows.append({
                "model": key,
                "model_display": MODEL_DISPLAY[key],
                "ordering_accuracy": row["ordering_accuracy_mean"],
                "negation_invariance": row["negation_invariance_mean"],
                "topic_coherence": row["topic_coherence_mean"],
            })

    if not rows:
        return None

    # Deduplicate (SBERT and SBERT-EUCLIDEAN both map to sbert)
    df = pd.DataFrame(rows).drop_duplicates(subset=["model"], keep="first")
    return df


# ---------------------------------------------------------------------------
# Narrative report
# ---------------------------------------------------------------------------

def generate_narrative_report(
    combined: pd.DataFrame,
    per_clone_rankings: dict[str, pd.DataFrame],
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    benchmark: pd.DataFrame | None,
    outpath: Path,
):
    """Write structured analysis report."""
    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)

    def subsection(title):
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    lines.append("Experiment 1 Analysis: Alpha Sweep Compilation")
    lines.append("(Generation 2: top5impact_n4, 3 clone conditions x 10 models)")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # --- Data overview ---
    section("1. Data Overview")
    n_models = combined["model"].nunique()
    n_clones = combined["clone_type"].nunique()
    n_alphas = combined["alpha"].nunique()
    lines.append(f"  Models: {n_models}")
    lines.append(f"  Clone conditions: {n_clones} ({', '.join(CLONE_DISPLAY[c] for c in CLONE_ORDER)})")
    lines.append(f"  Alpha values: {n_alphas}")
    lines.append(f"  Baseline (no CRW):")
    lines.append(f"    Jaccard:  {baselines['jaccard']:.4f}")
    lines.append(f"    Spearman: {baselines['spearman']:.4f}")
    lines.append(f"    Kendall:  {baselines['kendall']:.4f}")

    # --- Per-clone summaries ---
    section("2. Per-Clone Condition Results")
    for clone in CLONE_ORDER:
        df = per_clone_rankings[clone]
        subsection(f"Clone: {CLONE_DISPLAY[clone]}")

        lines.append(f"  Top 3 models (by composite rank):")
        for i, (_, row) in enumerate(df.head(3).iterrows()):
            lines.append(
                f"    {i+1}. {row['model']:22s}  "
                f"Jaccard={row['max_jaccard']:.4f}  "
                f"Spearman={row['max_spearman']:.4f}  "
                f"Kendall={row['max_kendall']:.4f}  "
                f"optimal_alpha={row['optimal_alpha_jaccard']:.1f}  "
                f"95%-interval={row['alpha_95_interval']}"
            )

        lines.append(f"\n  Full ranking:")
        lines.append(f"    {'Model':22s}  {'Jac':>7s}  {'Spe':>7s}  {'Ken':>7s}  "
                     f"{'rJ':>3s}  {'rS':>3s}  {'rK':>3s}  {'comp':>5s}  "
                     f"{'opt_a':>5s}  {'95% interval':>14s}")
        for _, row in df.iterrows():
            lines.append(
                f"    {row['model']:22s}  "
                f"{row['max_jaccard']:7.4f}  {row['max_spearman']:7.4f}  "
                f"{row['max_kendall']:7.4f}  "
                f"{row['rank_jaccard']:3d}  {row['rank_spearman']:3d}  "
                f"{row['rank_kendall']:3d}  "
                f"{row['composite_rank']:5.1f}  "
                f"{row['optimal_alpha_jaccard']:5.1f}  "
                f"{row['alpha_95_interval']:>14s}"
            )

    # --- Aggregated ranking ---
    section("3. Aggregated Model Ranking")
    lines.append("  (Average composite rank across all 3 clone conditions)")
    lines.append("")
    lines.append(f"    {'Rank':>4s}  {'Model':22s}  {'Avg Jaccard':>11s}  "
                 f"{'Avg Spearman':>12s}  {'Avg Kendall':>11s}  "
                 f"{'Avg Comp Rank':>13s}")
    for _, row in aggregated_ranking.iterrows():
        lines.append(
            f"    {row['rank']:4.0f}  {row['model']:22s}  "
            f"{row['avg_max_jaccard']:11.4f}  "
            f"{row['avg_max_spearman']:12.4f}  "
            f"{row['avg_max_kendall']:11.4f}  "
            f"{row['avg_composite_rank']:13.1f}"
        )

    # --- Peaked vs stable analysis ---
    section("4. Peaked vs Stable Model Analysis")
    lines.append("")
    lines.append("  Classification based on mean alpha_95_interval width across clones:")
    lines.append("    Peaked  (width < 0.3): Sharp distance structure, detects clones")
    lines.append("                           at a specific threshold.")
    lines.append("    Moderate (0.3-0.8):    Good discrimination with some tolerance.")
    lines.append("    Broad   (width >= 0.8): Mushy distance structure, CRW acts")
    lines.append("                            uniformly across thresholds.")
    lines.append("")

    # Compute mean interval width per model across clones
    model_widths = {}
    model_max_jaccards = {}
    for model_display in aggregated_ranking["model"]:
        widths = []
        max_js = []
        for clone, df in per_clone_rankings.items():
            row = df[df["model"] == model_display]
            if not row.empty:
                widths.append(row["alpha_95_width"].values[0])
                max_js.append(row["max_jaccard"].values[0])
        model_widths[model_display] = np.mean(widths) if widths else 0
        model_max_jaccards[model_display] = np.mean(max_js) if max_js else 0

    median_jaccard = np.median(list(model_max_jaccards.values()))

    peaked = []
    moderate = []
    broad = []
    for model_display in aggregated_ranking["model"]:
        w = model_widths[model_display]
        j = model_max_jaccards[model_display]
        entry = f"{model_display:22s}  width={w:.1f}  avg_max_jaccard={j:.4f}"
        if w < 0.3:
            peaked.append(entry)
        elif w < 0.8:
            moderate.append(entry)
        else:
            broad.append(entry)

    lines.append(f"  PEAKED (width < 0.3) — {len(peaked)} models:")
    for e in peaked:
        lines.append(f"    {e}")
    if not peaked:
        lines.append("    (none)")

    lines.append(f"\n  MODERATE (0.3 <= width < 0.8) — {len(moderate)} models:")
    for e in moderate:
        lines.append(f"    {e}")
    if not moderate:
        lines.append("    (none)")

    lines.append(f"\n  BROAD (width >= 0.8) — {len(broad)} models:")
    for e in broad:
        lines.append(f"    {e}")
    if not broad:
        lines.append("    (none)")

    lines.append("")
    lines.append("  Interpretation:")
    lines.append("  - Peaked models have clear distance structure: clones cluster at")
    lines.append("    a specific distance, and CRW finds them sharply. The peak IS the")
    lines.append("    signal — the model found the clones at a specific threshold.")
    lines.append("  - Broad models with low max Jaccard have mushy distance structure:")
    lines.append("    distances between clones and non-clones are similar, so CRW acts")
    lines.append("    uniformly across all thresholds without strong correction.")
    lines.append("  - Broad models with HIGH max Jaccard are robust: they correct well")
    lines.append("    across a wide alpha range, making alpha choice less critical.")

    # --- Benchmark correlation ---
    if benchmark is not None:
        section("5. Fake Benchmark Correlation")
        lines.append("")
        lines.append("  Do models that score high on the fake benchmark also achieve")
        lines.append("  better CRW correction? Correlating all 3 benchmark metrics")
        lines.append("  against all sweep performance metrics.")
        lines.append("")
        lines.append("  Benchmark metrics (direction):")
        lines.append("    ordering_accuracy  — higher is better")
        lines.append("    negation_invariance — lower is better (ratio)")
        lines.append("    topic_coherence     — lower is better (ratio)")
        lines.append("")

        # Merge benchmark with aggregated ranking
        display_to_key = {v: k for k, v in MODEL_DISPLAY.items()}
        merged_rows = []
        for _, agg_row in aggregated_ranking.iterrows():
            key = display_to_key.get(agg_row["model"])
            bench_row = benchmark[benchmark["model"] == key]
            if not bench_row.empty:
                merged_rows.append({
                    "model": agg_row["model"],
                    "avg_composite_rank": agg_row["avg_composite_rank"],
                    "avg_max_jaccard": agg_row["avg_max_jaccard"],
                    "avg_max_spearman": agg_row["avg_max_spearman"],
                    "avg_max_kendall": agg_row["avg_max_kendall"],
                    "ordering_accuracy": bench_row["ordering_accuracy"].values[0],
                    "negation_invariance": bench_row["negation_invariance"].values[0],
                    "topic_coherence": bench_row["topic_coherence"].values[0],
                })

        if len(merged_rows) >= 3:
            merged = pd.DataFrame(merged_rows)

            bench_metrics = ["ordering_accuracy", "negation_invariance", "topic_coherence"]
            sweep_metrics = ["avg_max_jaccard", "avg_max_spearman", "avg_max_kendall",
                             "avg_composite_rank"]

            # Compute full correlation matrix
            corr_rows = []
            for bm in bench_metrics:
                for sm in sweep_metrics:
                    rho, p = stats.spearmanr(merged[bm], merged[sm])
                    corr_rows.append({
                        "benchmark_metric": bm,
                        "sweep_metric": sm,
                        "spearman_rho": rho,
                        "p_value": p,
                    })
            corr_df = pd.DataFrame(corr_rows)

            # Save CSV
            corr_csv_path = outpath.parent / "benchmark_sweep_correlation.csv"
            corr_df.to_csv(corr_csv_path, index=False)
            print(f"  -> {corr_csv_path.name}")

            # Save heatmap
            rho_matrix = corr_df.pivot(
                index="benchmark_metric", columns="sweep_metric", values="spearman_rho"
            )
            p_matrix = corr_df.pivot(
                index="benchmark_metric", columns="sweep_metric", values="p_value"
            )
            # Reorder rows and columns
            rho_matrix = rho_matrix.loc[bench_metrics, sweep_metrics]
            p_matrix = p_matrix.loc[bench_metrics, sweep_metrics]

            # Build annotation strings: "rho\n(p=...)" with significance stars
            annot = np.empty_like(rho_matrix, dtype=object)
            for i, bm in enumerate(bench_metrics):
                for j, sm in enumerate(sweep_metrics):
                    r = rho_matrix.loc[bm, sm]
                    p = p_matrix.loc[bm, sm]
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    annot[i, j] = f"{r:.3f}{stars}\n(p={p:.3f})"

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(
                rho_matrix.astype(float), annot=annot, fmt="",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                xticklabels=[s.replace("avg_", "").replace("_", " ") for s in sweep_metrics],
                yticklabels=bench_metrics,
            )
            ax.set_title("Benchmark vs Sweep: Spearman Correlations")
            fig.tight_layout()
            plot_path = outpath.parent / "benchmark_sweep_correlation.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  -> {plot_path.name}")

            # Write correlation matrix to report
            lines.append("  Spearman correlation matrix (benchmark × sweep):")
            lines.append("")
            header = f"  {'':24s}" + "".join(f"  {s:>16s}" for s in sweep_metrics)
            lines.append(header)
            for bm in bench_metrics:
                row_str = f"  {bm:24s}"
                for sm in sweep_metrics:
                    r = rho_matrix.loc[bm, sm]
                    p = p_matrix.loc[bm, sm]
                    sig = "*" if p < 0.05 else " "
                    row_str += f"  {r:>+7.3f} (p={p:.3f}){sig}"
                lines.append(row_str)
            lines.append("")
            lines.append("  * = significant at p < 0.05")
            lines.append(f"  Full matrix saved to: {corr_csv_path.name}")
            lines.append(f"  Heatmap saved to: {plot_path.name}")
            lines.append("")

            # Model table with all benchmark metrics
            lines.append(f"  {'Model':22s}  {'Ord Acc':>8s}  {'Neg Inv':>8s}  {'Top Coh':>8s}  "
                         f"{'Jaccard':>8s}  {'Spearman':>8s}  {'Kendall':>8s}  {'Rank':>5s}")
            merged_sorted = merged.sort_values("ordering_accuracy", ascending=False)
            for _, row in merged_sorted.iterrows():
                lines.append(
                    f"  {row['model']:22s}  "
                    f"{row['ordering_accuracy']:7.1f}%  "
                    f"{row['negation_invariance']:8.3f}  "
                    f"{row['topic_coherence']:8.3f}  "
                    f"{row['avg_max_jaccard']:8.4f}  "
                    f"{row['avg_max_spearman']:8.4f}  "
                    f"{row['avg_max_kendall']:8.4f}  "
                    f"{row['avg_composite_rank']:5.1f}"
                )
        else:
            lines.append("  Not enough matched models for correlation analysis.")
    else:
        section("5. Fake Benchmark Correlation")
        lines.append("  Benchmark CSV not available — skipping correlation analysis.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("End of report")
    lines.append("=" * 80)

    outpath.write_text("\n".join(lines))
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Gen2 alpha sweep CSVs...")
    combined = load_gen2_data()

    print("Extracting baselines...")
    baselines = extract_baselines(combined)
    print(f"  Jaccard:  {baselines['jaccard']:.4f}")
    print(f"  Spearman: {baselines['spearman']:.4f}")
    print(f"  Kendall:  {baselines['kendall']:.4f}")

    # --- Per-metric comparison plots (9) ---
    print("\nGenerating per-metric comparison plots...")
    for clone in CLONE_ORDER:
        for metric_key in METRICS:
            outpath = OUTPUT_DIR / f"{clone}_{metric_key}_vs_alpha.png"
            plot_metric_comparison(combined, baselines, clone, metric_key, outpath)

    # --- Ranking tables (3 per-clone + 1 aggregated) ---
    print("\nComputing ranking tables...")
    per_clone_rankings = {}
    for clone in CLONE_ORDER:
        ranking = compute_ranking_table(combined, clone)
        per_clone_rankings[clone] = ranking
        csv_path = OUTPUT_DIR / f"{clone}_model_ranking.csv"
        ranking.to_csv(csv_path, index=False)
        print(f"  -> {csv_path.name}")

    aggregated = compute_aggregated_ranking(per_clone_rankings)
    agg_path = OUTPUT_DIR / "aggregated_model_ranking.csv"
    aggregated.to_csv(agg_path, index=False)
    print(f"  -> {agg_path.name}")

    # Print aggregated ranking
    print("\n" + "=" * 70)
    print("AGGREGATED MODEL RANKING")
    print("=" * 70)
    print(aggregated.to_string(index=False, float_format="%.3f"))

    # --- Ranking bar charts (3 + 1) ---
    print("\nGenerating ranking bar charts...")
    for clone in CLONE_ORDER:
        title = f"{CLONE_DISPLAY[clone]} \u2014 Model Ranking"
        outpath = OUTPUT_DIR / f"{clone}_model_ranking.png"
        plot_ranking_bars(per_clone_rankings[clone], baselines, title, outpath)

    plot_ranking_bars(
        aggregated, baselines,
        "Aggregated Model Ranking (across all clone conditions)",
        OUTPUT_DIR / "aggregated_model_ranking.png",
        jaccard_col="avg_max_jaccard",
        spearman_col="avg_max_spearman",
        kendall_col="avg_max_kendall",
    )

    # --- Narrative report ---
    print("\nGenerating narrative report...")
    benchmark = load_benchmark_scores()
    generate_narrative_report(
        combined, per_clone_rankings, aggregated, baselines, benchmark,
        OUTPUT_DIR / "exp1_narrative_report.txt",
    )

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
