# -*- coding: utf-8 -*-
"""
Compile IC2S2 alpha sweep results: 6 clone conditions x 10 models.

Reads all Gen2 (top5impact_n4) alpha sweep CSVs and produces:
  - Per-metric comparison plots (18): all models on one axes per (clone, metric)
  - Model ranking tables (7 CSVs): per-clone + aggregated
  - Model ranking bar charts (7): grouped bars per model
  - Peak Jaccard heatmap: 10x6 model x clone type overview
  - Benchmark correlation scatter: ordering accuracy vs CRW Jaccard
  - Abstract-ready numbers CSV
  - Narrative report with abstract-ready stats

Usage:
    python scripts/compile_ic2s2_alpha_sweep.py
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
RESULTS_DIR = Path("experiment_results/alpha_sweep_results")
BENCHMARK_CSV = Path("experiment_results/model_benchmark/model_comparison.csv")
OUTPUT_DIR = Path("experiment_results/ic2s2_compilation")

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

CLONE_ORDER = [
    "easy_paraphrase", "hard_paraphrase",
    "negation_easy", "negation_hard",
    "mixed", "perfect_mix",
]
CLONE_DISPLAY = {
    "easy_paraphrase": "Easy Paraphrase",
    "hard_paraphrase": "Hard Paraphrase",
    "negation_easy": "Negation + Easy Para.",
    "negation_hard": "Negation + Hard Para.",
    "mixed": "Mixed (Easy)",
    "perfect_mix": "Perfect Mix",
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

# Ranking bar chart colors
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
# Plot — Per-metric comparison
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
# Plot — Peak Jaccard heatmap (model x clone type)
# ---------------------------------------------------------------------------

def plot_peak_jaccard_heatmap(
    per_clone_rankings: dict[str, pd.DataFrame],
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    outpath: Path,
):
    """10x6 heatmap of peak Jaccard values, models sorted by aggregated rank."""
    sns.set_theme(style="white")

    # Build pivot: rows = models (sorted by rank), cols = clone types
    model_order = aggregated_ranking["model"].tolist()

    rows = []
    for model in model_order:
        row = {"model": model}
        for clone in CLONE_ORDER:
            df = per_clone_rankings[clone]
            model_row = df[df["model"] == model]
            if not model_row.empty:
                row[clone] = model_row["max_jaccard"].values[0]
            else:
                row[clone] = np.nan
        rows.append(row)

    pivot = pd.DataFrame(rows).set_index("model")
    pivot.columns = [CLONE_DISPLAY[c] for c in CLONE_ORDER]

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=baselines["jaccard"] * 0.95,
        vmax=pivot.max().max() * 1.02,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Peak Jaccard (best \u03b1)", "shrink": 0.8},
        annot_kws={"size": 10},
    )

    ax.set_title(
        f"Peak CRW Jaccard per Model \u00d7 Clone Type\n"
        f"(baseline = {baselines['jaccard']:.3f}, models sorted by aggregated rank)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Benchmark correlation scatter
# ---------------------------------------------------------------------------

def plot_benchmark_scatter(
    aggregated_ranking: pd.DataFrame,
    benchmark: pd.DataFrame,
    baselines: dict[str, float],
    outpath: Path,
):
    """Scatter plot: benchmark ordering accuracy vs CRW avg Jaccard."""
    display_to_key = {v: k for k, v in MODEL_DISPLAY.items()}

    points = []
    for _, agg_row in aggregated_ranking.iterrows():
        key = display_to_key.get(agg_row["model"])
        bench_row = benchmark[benchmark["model"] == key]
        if not bench_row.empty:
            points.append({
                "model": agg_row["model"],
                "ordering_accuracy": bench_row["ordering_accuracy"].values[0],
                "avg_max_jaccard": agg_row["avg_max_jaccard"],
            })

    if len(points) < 3:
        print("  WARNING: Not enough points for benchmark scatter")
        return

    df = pd.DataFrame(points)

    rho, p_val = stats.spearmanr(df["ordering_accuracy"], df["avg_max_jaccard"])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        df["ordering_accuracy"], df["avg_max_jaccard"],
        s=100, c=[MODEL_COLORS[display_to_key[m]] for m in df["model"]],
        edgecolors="black", linewidths=0.5, zorder=5,
    )

    # Label each point
    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (row["ordering_accuracy"], row["avg_max_jaccard"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9, ha="left",
        )

    # Regression line
    z = np.polyfit(df["ordering_accuracy"], df["avg_max_jaccard"], 1)
    x_line = np.linspace(df["ordering_accuracy"].min() - 1, df["ordering_accuracy"].max() + 1, 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.6, linewidth=1)

    # Baseline
    ax.axhline(
        baselines["jaccard"], color="black", linestyle=":", linewidth=1, alpha=0.5,
        label=f"Baseline Jaccard: {baselines['jaccard']:.3f}",
    )

    ax.set_xlabel("Benchmark Ordering Accuracy (%)", fontsize=12)
    ax.set_ylabel("Avg Peak CRW Jaccard (across 6 clone types)", fontsize=12)
    ax.set_title(
        f"Embedding Model Benchmark vs CRW Effectiveness\n"
        f"Spearman \u03c1 = {rho:.3f} (p = {p_val:.4f})",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")

    return rho, p_val


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
            })

    if not rows:
        return None

    # Deduplicate (SBERT and SBERT-EUCLIDEAN both map to sbert)
    df = pd.DataFrame(rows).drop_duplicates(subset=["model"], keep="first")
    return df


# ---------------------------------------------------------------------------
# Plot — Winner showcase (best vs worst model across clone types)
# ---------------------------------------------------------------------------

# Clone types ordered by detection difficulty (easy → hard)
_CLONE_DIFFICULTY_ORDER = [
    "easy_paraphrase", "negation_easy", "mixed",
    "perfect_mix", "hard_paraphrase", "negation_hard",
]
_CLONE_DIFFICULTY_COLORS = {
    "easy_paraphrase": "#4CAF50",   # green — easiest
    "negation_easy": "#8BC34A",     # light green
    "mixed": "#FFC107",             # amber
    "perfect_mix": "#FF9800",       # orange
    "hard_paraphrase": "#F44336",   # red
    "negation_hard": "#B71C1C",     # dark red — hardest
}


def plot_winner_showcase(
    combined: pd.DataFrame,
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    outpath: Path,
):
    """Two-panel plot: best model vs worst model, alpha curves across all clone types.

    Shows WHY the best model wins: sharp peaks at a consistent alpha,
    while the worst model has flat, barely-above-baseline curves.
    """
    display_to_key = {v: k for k, v in MODEL_DISPLAY.items()}
    best_display = aggregated_ranking.iloc[0]["model"]
    worst_display = aggregated_ranking.iloc[-1]["model"]
    best_key = display_to_key[best_display]
    worst_key = display_to_key[worst_display]

    sns.set_theme(style="whitegrid")
    fig, (ax_best, ax_worst) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    baseline = baselines["jaccard"]
    available = [c for c in _CLONE_DIFFICULTY_ORDER if c in combined["clone_type"].unique()]

    for ax, model_key, model_display, panel_title in [
        (ax_best, best_key, best_display, f"Best Model: {best_display}"),
        (ax_worst, worst_key, worst_display, f"Worst Model: {worst_display}"),
    ]:
        for clone in available:
            data = combined[
                (combined["model"] == model_key) & (combined["clone_type"] == clone)
            ].sort_values("alpha")
            if data.empty:
                continue

            color = _CLONE_DIFFICULTY_COLORS[clone]
            label = CLONE_DISPLAY[clone]

            ax.plot(
                data["alpha"], data["crw_jaccard_mean"],
                color=color, linewidth=2, label=label,
                marker="o", markersize=3,
            )

            # Mark optimal alpha
            best_idx = data["crw_jaccard_mean"].idxmax()
            ax.plot(
                data.loc[best_idx, "alpha"],
                data.loc[best_idx, "crw_jaccard_mean"],
                marker="*", color=color, markersize=14, zorder=5,
            )

        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
        ax.text(
            2.8, baseline + 0.003, f"Baseline: {baseline:.3f}",
            fontsize=9, ha="right", va="bottom", color="black", alpha=0.7,
        )
        ax.set_xlabel(r"Alpha ($\alpha$)", fontsize=12)
        ax.set_title(panel_title, fontsize=14, fontweight="bold")

    ax_best.set_ylabel("Jaccard Similarity (mean)", fontsize=12)

    # Single legend below both panels
    handles, labels = ax_best.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(available),
        fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=True,
    )

    fig.suptitle(
        "CRW Correction: Best vs Worst Embedding Model\n"
        "(\u2605 = optimal \u03b1; clone types ordered by detection difficulty)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Model ranking dot plot
# ---------------------------------------------------------------------------

def plot_model_ranking_dot(
    per_clone_rankings: dict[str, pd.DataFrame],
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    outpath: Path,
):
    """Horizontal dot plot: models sorted by avg Jaccard, whiskers = min/max across clones."""
    sns.set_theme(style="whitegrid")

    models = aggregated_ranking.sort_values("rank")["model"].tolist()
    n = len(models)

    # Compute per-model min/max Jaccard across clone types
    model_stats = {}
    for model in models:
        jaccards = []
        for clone, df in per_clone_rankings.items():
            row = df[df["model"] == model]
            if not row.empty:
                jaccards.append(row["max_jaccard"].values[0])
        model_stats[model] = {
            "mean": np.mean(jaccards),
            "min": np.min(jaccards),
            "max": np.max(jaccards),
        }

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.55)))

    y = np.arange(n)
    baseline = baselines["jaccard"]

    # Baseline vertical line
    ax.axvline(baseline, color="black", linestyle="--", linewidth=1.5, alpha=0.5,
               label=f"Baseline (no CRW): {baseline:.3f}")

    for i, model in enumerate(models):
        s = model_stats[model]
        # Whisker: min to max
        ax.plot(
            [s["min"], s["max"]], [i, i],
            color="#90CAF9", linewidth=3, solid_capstyle="round", zorder=3,
        )
        # Dot: mean
        is_best = (i == 0)
        ax.scatter(
            s["mean"], i,
            s=144 if is_best else 81,
            color="#1565C0" if is_best else "#2196F3",
            edgecolors="black" if is_best else "none",
            linewidths=1.5 if is_best else 0,
            zorder=5,
        )
        # Value label
        ax.text(
            s["max"] + 0.005, i,
            f"{s['mean']:.3f}",
            va="center", fontsize=10, fontweight="bold" if is_best else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Peak Jaccard Similarity (higher = better CRW correction)", fontsize=11)
    ax.set_title(
        "Model Ranking: Average Peak Jaccard\n"
        "(dot = mean across 6 clone types, bar = min\u2013max range)",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Clone difficulty gradient
# ---------------------------------------------------------------------------

def plot_clone_difficulty_gradient(
    per_clone_rankings: dict[str, pd.DataFrame],
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    outpath: Path,
):
    """Bar chart for the best model: peak Jaccard per clone type, sorted by difficulty."""
    best_model = aggregated_ranking.iloc[0]["model"]
    baseline = baselines["jaccard"]

    available = [c for c in _CLONE_DIFFICULTY_ORDER if c in per_clone_rankings]

    clone_labels = []
    jaccards = []
    colors = []
    for clone in available:
        df = per_clone_rankings[clone]
        row = df[df["model"] == best_model]
        if not row.empty:
            clone_labels.append(CLONE_DISPLAY[clone])
            jaccards.append(row["max_jaccard"].values[0])
            colors.append(_CLONE_DIFFICULTY_COLORS[clone])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(clone_labels))
    bars = ax.bar(x, jaccards, color=colors, edgecolor="white", linewidth=0.5)

    # Baseline line
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(
        len(clone_labels) - 0.5, baseline + 0.003,
        f"Baseline: {baseline:.3f}", ha="right", va="bottom",
        fontsize=9, color="black", alpha=0.7,
    )

    # Value labels on bars
    for bar, val in zip(bars, jaccards):
        recovery = (val - baseline) / (1.0 - baseline) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}\n({recovery:.0f}% recovery)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(clone_labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Peak Jaccard Similarity (best \u03b1)", fontsize=11)
    ax.set_title(
        f"CRW Effectiveness by Clone Difficulty ({best_model})\n"
        f"(% recovery = fraction of distortion corrected by CRW)",
        fontsize=13,
    )
    ax.set_ylim(bottom=baseline * 0.92)

    # Gradient arrow annotation
    ax.annotate(
        "Easier to detect \u2192 Harder to detect",
        xy=(0.5, 0.02), xycoords="axes fraction",
        fontsize=10, ha="center", color="gray", fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — CRW recovery plot
# ---------------------------------------------------------------------------

def plot_crw_recovery(
    per_clone_rankings: dict[str, pd.DataFrame],
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    outpath: Path,
):
    """Per clone type: % of Jaccard distortion recovered by CRW for top 3 models."""
    baseline = baselines["jaccard"]
    top3 = aggregated_ranking.head(3)["model"].tolist()

    available = [c for c in _CLONE_DIFFICULTY_ORDER if c in per_clone_rankings]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    n_clones = len(available)
    n_models = len(top3)
    width = 0.22
    x = np.arange(n_clones)

    bar_colors = ["#1565C0", "#42A5F5", "#90CAF9"]  # dark to light blue for top 3

    for j, model in enumerate(top3):
        recoveries = []
        peak_jaccards = []
        for clone in available:
            df = per_clone_rankings[clone]
            row = df[df["model"] == model]
            if not row.empty:
                peak_j = row["max_jaccard"].values[0]
                recovery = (peak_j - baseline) / (1.0 - baseline) * 100
                recoveries.append(recovery)
                peak_jaccards.append(peak_j)
            else:
                recoveries.append(0)
                peak_jaccards.append(baseline)

        offset = (j - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, recoveries, width,
            label=model, color=bar_colors[j], edgecolor="white", linewidth=0.5,
        )

        # Value labels
        for bar, rec, pj in zip(bars, recoveries, peak_jaccards):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rec:.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CLONE_DISPLAY[c] for c in available],
        fontsize=10, rotation=15, ha="right",
    )
    ax.set_ylabel("Distortion Recovered by CRW (%)", fontsize=12)
    ax.set_title(
        "CRW Mitigation: Fraction of Cloning Distortion Recovered\n"
        f"(top 3 models; baseline Jaccard = {baseline:.3f}, perfect = 1.000)",
        fontsize=13,
    )
    ax.legend(fontsize=11, title="Model", title_fontsize=11)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Hero plot: single "CRW works" demonstration
# ---------------------------------------------------------------------------

def plot_hero(
    combined: pd.DataFrame,
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    outpath: Path,
    clone_type: str = "mixed",
):
    """Single-panel plot: best model's alpha curve vs baseline.

    The simplest possible demonstration that CRW corrects clone distortion.
    Shows: baseline (problem), CRW curve (solution), optimal alpha (parameter).
    """
    display_to_key = {v: k for k, v in MODEL_DISPLAY.items()}
    best_display = aggregated_ranking.iloc[0]["model"]
    best_key = display_to_key[best_display]
    baseline = baselines["jaccard"]

    data = combined[
        (combined["model"] == best_key) & (combined["clone_type"] == clone_type)
    ].sort_values("alpha")

    if data.empty:
        print(f"  WARNING: No data for {best_key} x {clone_type}")
        return

    best_idx = data["crw_jaccard_mean"].idxmax()
    opt_alpha = data.loc[best_idx, "alpha"]
    peak_jaccard = data.loc[best_idx, "crw_jaccard_mean"]
    recovery_pct = (peak_jaccard - baseline) / (1.0 - baseline) * 100

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # CRW curve
    ax.plot(
        data["alpha"], data["crw_jaccard_mean"],
        color="#1565C0", linewidth=2.5, zorder=4,
        label=f"CRW ({best_display})",
    )

    # Shade the improvement area between baseline and curve
    ax.fill_between(
        data["alpha"], baseline, data["crw_jaccard_mean"],
        where=data["crw_jaccard_mean"] > baseline,
        color="#1565C0", alpha=0.12, zorder=2,
    )

    # Baseline
    ax.axhline(baseline, color="#E53935", linestyle="--", linewidth=2, alpha=0.8,
               label=f"Baseline (no CRW): {baseline:.3f}", zorder=3)

    # Optimal alpha marker
    ax.plot(opt_alpha, peak_jaccard, marker="*", color="#1565C0",
            markersize=18, zorder=6)

    # Annotation: improvement arrow
    ax.annotate(
        f"  +{peak_jaccard - baseline:.3f}\n  ({recovery_pct:.0f}% of distortion\n  recovered)",
        xy=(opt_alpha, peak_jaccard),
        xytext=(opt_alpha + 0.6, peak_jaccard - 0.01),
        fontsize=11, fontweight="bold", color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
        va="center",
    )

    # Annotation: optimal alpha
    ax.annotate(
        f"optimal \u03b1 = {opt_alpha}",
        xy=(opt_alpha, baseline),
        xytext=(opt_alpha, baseline - 0.015),
        fontsize=10, ha="center", color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    )

    # Perfect similarity reference
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(2.9, 1.002, "Perfect (1.0)", fontsize=8, ha="right", color="gray", alpha=0.5)

    ax.set_xlabel(r"CRW threshold parameter $\alpha$", fontsize=12)
    ax.set_ylabel("Jaccard Similarity\n(original vs CRW-corrected recommendations)", fontsize=11)
    ax.set_title(
        f"CRW Corrects Clone-Induced Recommendation Distortion\n"
        f"({CLONE_DISPLAY[clone_type]} clones, {best_display} embeddings, "
        f"5 cloned questions \u00d7 4 clones = 20 added)",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(0, 3.1)
    ax.set_ylim(bottom=baseline - 0.03)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Abstract-ready numbers CSV
# ---------------------------------------------------------------------------

def generate_abstract_numbers(
    per_clone_rankings: dict[str, pd.DataFrame],
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    benchmark_rho: float | None,
    benchmark_p: float | None,
    outpath: Path,
):
    """Write a CSV with key numbers for quick reference when writing the abstract."""
    rows = []

    def add(metric, value, context):
        rows.append({"metric": metric, "value": value, "context": context})

    # Baselines
    add("baseline_jaccard", f"{baselines['jaccard']:.4f}", "No CRW (distorted by clones)")
    add("baseline_spearman", f"{baselines['spearman']:.4f}", "No CRW (distorted)")
    add("baseline_kendall", f"{baselines['kendall']:.4f}", "No CRW (distorted)")

    # Best model overall
    best = aggregated_ranking.iloc[0]
    add("best_model", best["model"], f"Rank 1 across all 6 clone types")
    add("best_model_jaccard", f"{best['avg_max_jaccard']:.4f}",
        f"{best['model']} avg peak Jaccard across 6 clones")
    improvement = best["avg_max_jaccard"] - baselines["jaccard"]
    add("best_improvement_abs", f"{improvement:.4f}",
        f"Absolute Jaccard improvement ({best['model']})")
    add("best_improvement_pct", f"{improvement / baselines['jaccard'] * 100:.1f}%",
        f"Relative Jaccard improvement ({best['model']})")

    # Worst model
    worst = aggregated_ranking.iloc[-1]
    add("worst_model", worst["model"], f"Rank {int(worst['rank'])} (last)")
    add("worst_model_jaccard", f"{worst['avg_max_jaccard']:.4f}",
        f"{worst['model']} avg peak Jaccard")

    # Spread
    add("model_spread", f"{best['avg_max_jaccard'] - worst['avg_max_jaccard']:.4f}",
        "Jaccard difference between best and worst model")

    # Per clone type best
    for clone in CLONE_ORDER:
        df = per_clone_rankings[clone]
        top = df.iloc[0]
        add(f"best_{clone}", top["model"],
            f"Best model for {CLONE_DISPLAY[clone]}")
        add(f"best_{clone}_jaccard", f"{top['max_jaccard']:.4f}",
            f"Peak Jaccard for {CLONE_DISPLAY[clone]} ({top['model']})")

    # Benchmark correlation
    if benchmark_rho is not None:
        add("benchmark_spearman_rho", f"{benchmark_rho:.3f}",
            "Ordering accuracy vs avg CRW Jaccard")
        add("benchmark_p_value", f"{benchmark_p:.4f}",
            "p-value of benchmark correlation")

    # Summary
    add("n_models", "10", "Embedding models tested")
    add("n_clone_types", "6", "Clone conditions")
    add("n_alphas", "21", "Alpha values swept")
    add("n_questions_cloned", "5", "Worst-case questions (top impact)")
    add("n_clones_per_question", "4", "Clones per question per condition")

    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Narrative report
# ---------------------------------------------------------------------------

def generate_narrative_report(
    combined: pd.DataFrame,
    per_clone_rankings: dict[str, pd.DataFrame],
    aggregated_ranking: pd.DataFrame,
    baselines: dict[str, float],
    benchmark: pd.DataFrame | None,
    benchmark_rho: float | None,
    benchmark_p: float | None,
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

    lines.append("IC2S2 Alpha Sweep Compilation")
    lines.append("(10 models x 6 clone conditions, top5impact_n4)")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # --- Data overview ---
    section("1. Data Overview")
    n_models = combined["model"].nunique()
    n_clones = combined["clone_type"].nunique()
    n_alphas = combined["alpha"].nunique()
    lines.append(f"  Models: {n_models}")
    lines.append(f"  Clone conditions: {n_clones} ({', '.join(CLONE_DISPLAY[c] for c in CLONE_ORDER if c in combined['clone_type'].unique())})")
    lines.append(f"  Alpha values: {n_alphas}")
    lines.append(f"  Questions cloned: 5 (highest-impact: Q32214, Q32261, Q32228, Q32234, Q32222)")
    lines.append(f"  Clones per question: 4")
    lines.append(f"  Baseline (no CRW):")
    lines.append(f"    Jaccard:  {baselines['jaccard']:.4f}")
    lines.append(f"    Spearman: {baselines['spearman']:.4f}")
    lines.append(f"    Kendall:  {baselines['kendall']:.4f}")

    # --- Per-clone summaries ---
    section("2. Per-Clone Condition Results")
    for clone in CLONE_ORDER:
        if clone not in per_clone_rankings:
            continue
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
    lines.append(f"  (Average composite rank across all {len(per_clone_rankings)} clone conditions)")
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
    lines.append("    a specific distance, and CRW finds them sharply.")
    lines.append("  - Broad models with low max Jaccard have mushy distance structure:")
    lines.append("    distances between clones and non-clones are similar.")
    lines.append("  - Broad models with HIGH max Jaccard are robust: they correct well")
    lines.append("    across a wide alpha range, making alpha choice less critical.")

    # --- Benchmark correlation ---
    if benchmark is not None:
        section("5. Benchmark Correlation")
        lines.append("")
        lines.append("  Do models that score high on topic similarity (ordering accuracy)")
        lines.append("  in the fake benchmark also achieve better CRW correction?")
        lines.append("")

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
                    "ordering_accuracy": bench_row["ordering_accuracy"].values[0],
                })

        if len(merged_rows) >= 3:
            merged = pd.DataFrame(merged_rows)

            rho_j, p_j = stats.spearmanr(
                merged["ordering_accuracy"], merged["avg_max_jaccard"]
            )
            rho_r, p_r = stats.spearmanr(
                merged["ordering_accuracy"], merged["avg_composite_rank"]
            )

            lines.append(f"  Spearman correlation (ordering_accuracy vs avg_max_jaccard):")
            lines.append(f"    rho = {rho_j:.3f}, p = {p_j:.4f}")
            lines.append(f"  Spearman correlation (ordering_accuracy vs avg_composite_rank):")
            lines.append(f"    rho = {rho_r:.3f}, p = {p_r:.4f}")
            lines.append(f"    (negative rho expected: higher accuracy -> lower/better rank)")
            lines.append("")

            lines.append(f"  {'Model':22s}  {'Ordering Acc':>12s}  {'Avg Jaccard':>11s}  "
                         f"{'Comp Rank':>9s}")
            merged_sorted = merged.sort_values("ordering_accuracy", ascending=False)
            for _, row in merged_sorted.iterrows():
                lines.append(
                    f"  {row['model']:22s}  "
                    f"{row['ordering_accuracy']:12.1f}%  "
                    f"{row['avg_max_jaccard']:11.4f}  "
                    f"{row['avg_composite_rank']:9.1f}"
                )
        else:
            lines.append("  Not enough matched models for correlation analysis.")
    else:
        section("5. Benchmark Correlation")
        lines.append("  Benchmark CSV not available — skipping.")

    # --- Abstract-ready numbers ---
    section("6. Abstract-Ready Numbers")
    lines.append("")
    best = aggregated_ranking.iloc[0]
    worst = aggregated_ranking.iloc[-1]
    improvement = best["avg_max_jaccard"] - baselines["jaccard"]

    lines.append(f"  Baseline Jaccard (distorted, no CRW):  {baselines['jaccard']:.4f}")
    lines.append(f"  Best model ({best['model']}):   avg Jaccard = {best['avg_max_jaccard']:.4f}  "
                 f"(+{improvement:.4f}, +{improvement / baselines['jaccard'] * 100:.1f}%)")
    lines.append(f"  Worst model ({worst['model']}): avg Jaccard = {worst['avg_max_jaccard']:.4f}")
    lines.append(f"  Model spread (best - worst):  {best['avg_max_jaccard'] - worst['avg_max_jaccard']:.4f}")
    lines.append("")

    lines.append("  Per clone type (best model, peak Jaccard):")
    for clone in CLONE_ORDER:
        if clone not in per_clone_rankings:
            continue
        top = per_clone_rankings[clone].iloc[0]
        imp = top["max_jaccard"] - baselines["jaccard"]
        lines.append(
            f"    {CLONE_DISPLAY[clone]:25s}  {top['model']:18s}  "
            f"Jaccard={top['max_jaccard']:.4f}  (+{imp:.4f})"
        )

    if benchmark_rho is not None:
        lines.append("")
        lines.append(f"  Benchmark correlation: rho={benchmark_rho:.3f}, p={benchmark_p:.4f}")

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

    print("Loading Gen2 alpha sweep CSVs (all 6 clone types)...")
    combined = load_gen2_data()

    print("Extracting baselines...")
    baselines = extract_baselines(combined)
    print(f"  Jaccard:  {baselines['jaccard']:.4f}")
    print(f"  Spearman: {baselines['spearman']:.4f}")
    print(f"  Kendall:  {baselines['kendall']:.4f}")

    # --- Per-metric comparison plots (18) ---
    print("\nGenerating per-metric comparison plots...")
    available_clones = [c for c in CLONE_ORDER if c in combined["clone_type"].unique()]
    for clone in available_clones:
        for metric_key in METRICS:
            outpath = OUTPUT_DIR / f"{clone}_{metric_key}_vs_alpha.png"
            plot_metric_comparison(combined, baselines, clone, metric_key, outpath)

    # --- Ranking tables (per-clone + aggregated) ---
    print("\nComputing ranking tables...")
    per_clone_rankings = {}
    for clone in available_clones:
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
    print("AGGREGATED MODEL RANKING (across 6 clone conditions)")
    print("=" * 70)
    print(aggregated.to_string(index=False, float_format="%.3f"))

    # --- Ranking bar charts ---
    print("\nGenerating ranking bar charts...")
    for clone in available_clones:
        title = f"{CLONE_DISPLAY[clone]} \u2014 Model Ranking"
        outpath = OUTPUT_DIR / f"{clone}_model_ranking.png"
        plot_ranking_bars(per_clone_rankings[clone], baselines, title, outpath)

    plot_ranking_bars(
        aggregated, baselines,
        "Aggregated Model Ranking (across all 6 clone conditions)",
        OUTPUT_DIR / "aggregated_model_ranking.png",
        jaccard_col="avg_max_jaccard",
        spearman_col="avg_max_spearman",
        kendall_col="avg_max_kendall",
    )

    # --- Heatmap ---
    print("\nGenerating peak Jaccard heatmap...")
    plot_peak_jaccard_heatmap(
        per_clone_rankings, aggregated, baselines,
        OUTPUT_DIR / "peak_jaccard_heatmap.png",
    )

    # --- Hero plot ---
    print("\nGenerating hero plot...")
    plot_hero(
        combined, aggregated, baselines,
        OUTPUT_DIR / "hero_crw_works.png",
        clone_type="mixed",
    )

    # --- Focused presentation-ready figures ---
    print("\nGenerating focused presentation figures...")
    plot_winner_showcase(
        combined, aggregated, baselines,
        OUTPUT_DIR / "winner_showcase.png",
    )
    plot_model_ranking_dot(
        per_clone_rankings, aggregated, baselines,
        OUTPUT_DIR / "model_ranking_dot.png",
    )
    plot_clone_difficulty_gradient(
        per_clone_rankings, aggregated, baselines,
        OUTPUT_DIR / "clone_difficulty_gradient.png",
    )
    plot_crw_recovery(
        per_clone_rankings, aggregated, baselines,
        OUTPUT_DIR / "crw_recovery.png",
    )

    # --- Benchmark scatter + correlation ---
    print("\nLoading benchmark scores...")
    benchmark = load_benchmark_scores()
    benchmark_rho = None
    benchmark_p = None
    if benchmark is not None:
        print("Generating benchmark correlation scatter...")
        result = plot_benchmark_scatter(
            aggregated, benchmark, baselines,
            OUTPUT_DIR / "benchmark_correlation_scatter.png",
        )
        if result:
            benchmark_rho, benchmark_p = result

    # --- Narrative report ---
    print("\nGenerating narrative report...")
    generate_narrative_report(
        combined, per_clone_rankings, aggregated, baselines,
        benchmark, benchmark_rho, benchmark_p,
        OUTPUT_DIR / "ic2s2_report.txt",
    )

    # --- Abstract numbers CSV ---
    print("\nGenerating abstract numbers CSV...")
    generate_abstract_numbers(
        per_clone_rankings, aggregated, baselines,
        benchmark_rho, benchmark_p,
        OUTPUT_DIR / "ic2s2_abstract_numbers.csv",
    )

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
