# -*- coding: utf-8 -*-
"""
Compile alpha sweep results across all embedding models and clone types.

Reads all alpha_sweep_*.csv files, extracts baseline metrics from worker CSVs,
and produces:
  - alpha_sweep_summary.csv         — per model x clone-type summary stats
  - alpha_sweep_model_summary.csv   — per-model ranking with optimal alpha range
  - alpha_curves_jaccard.png        — Jaccard vs alpha, faceted by model
  - alpha_curves_spearman.png       — Spearman vs alpha, faceted by model
  - model_comparison_heatmaps.png   — peak CRW / improvement heatmaps
  - optimal_alpha_heatmap.png       — optimal alpha stability across conditions
  - model_ranking.png               — grouped bar chart comparing models

Usage:
    python scripts/compile_alpha_sweep.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("experiment_results/exp1/model_alpha_sweep/top5impact")
OUTPUT_DIR = Path("experiment_results/exp1/model_alpha_sweep/compiled")

# ---------------------------------------------------------------------------
# Display names and ordering
# ---------------------------------------------------------------------------
MODEL_ORDER = ["e5", "e5_instruct", "jina_v3", "qwen3"]
MODEL_DISPLAY = {
    "e5": "E5",
    "e5_instruct": "E5-Instruct",
    "jina_v3": "Jina v3",
    "qwen3": "Qwen3",
}

CLONE_ORDER = [
    "negation_q32214_n5",
    "easy_paraphrase_q32214_n5",
    "hard_paraphrase_q32214_n5",
    "negation_easy_q32214_n5",
    "negation_hard_q32214_n5",
    "natural_mixed_q32214_n5",
]
CLONE_DISPLAY = {
    "negation_q32214_n5": "Negation",
    "easy_paraphrase_q32214_n5": "Easy Para.",
    "hard_paraphrase_q32214_n5": "Hard Para.",
    "negation_easy_q32214_n5": "Neg + Easy",
    "negation_hard_q32214_n5": "Neg + Hard",
    "natural_mixed_q32214_n5": "Mixed",
}

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
CLONE_COLORS = {
    "negation_q32214_n5": "#1f77b4",
    "easy_paraphrase_q32214_n5": "#ff7f0e",
    "hard_paraphrase_q32214_n5": "#bcbd22",
    "negation_easy_q32214_n5": "#2ca02c",
    "negation_hard_q32214_n5": "#9467bd",
    "natural_mixed_q32214_n5": "#7f7f7f",
}
CLONE_LINESTYLES = {
    "negation_q32214_n5": "--",
    "easy_paraphrase_q32214_n5": "-",
    "hard_paraphrase_q32214_n5": "-",
    "negation_easy_q32214_n5": "--",
    "negation_hard_q32214_n5": "--",
    "natural_mixed_q32214_n5": ":",
}

_COLOR_BAND = "#2196F3"  # blue for optimal alpha band

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filepath: Path) -> tuple[str, str]:
    """Extract (model, clone_type) from an alpha sweep CSV filename."""
    stem = filepath.stem
    parts = stem.split("_vs_", 1)
    config_a = parts[0]
    config_b = parts[1]

    model = config_a.replace("alpha_sweep_pipeline_", "").removesuffix("_ZH")

    suffix_pat = rf"_{re.escape(model)}_ZH_\d{{4}}_\d{{4}}_[a-f0-9]+$"
    clone_type = re.sub(suffix_pat, "", config_b)

    return model, clone_type


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_main_csvs() -> pd.DataFrame:
    """Load all alpha_sweep_*.csv files, excluding identical clones."""
    csv_files = sorted(
        f for f in RESULTS_DIR.glob("**/alpha_sweep_*.csv")
        if "old" not in f.relative_to(RESULTS_DIR).parts
    )
    if not csv_files:
        print(f"ERROR: No CSV files found in {RESULTS_DIR}")
        sys.exit(1)

    frames = []
    for f in csv_files:
        model, clone_type = parse_filename(f)
        # Filter out identical clones
        if "identical" in clone_type:
            continue
        df = pd.read_csv(f)
        df["model"] = model
        df["clone_type"] = clone_type
        match = re.search(r"_(\d{4})_(\d{4})_[a-f0-9]+\.csv$", f.name)
        df["_ts"] = match.group(1) + match.group(2) if match else "0000"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep the latest run per (model, clone_type, alpha)
    combined = combined.sort_values("_ts").drop_duplicates(
        subset=["model", "clone_type", "alpha"], keep="last"
    )
    combined = combined.drop(columns=["_ts"])
    return combined


def load_baseline_metrics() -> dict[tuple[str, str], dict]:
    """Read baseline metrics from worker CSVs (one per model x clone combo)."""
    baselines = {}
    for model in MODEL_ORDER:
        for clone in CLONE_ORDER:
            subfolder = RESULTS_DIR / f"alpha_sweep_pipeline_{model}_ZH_vs_{clone}_{model}_ZH"
            if subfolder.is_dir():
                worker_dirs = sorted(subfolder.glob("workers_*"))
            else:
                worker_dirs = sorted(RESULTS_DIR.glob(f"workers_{clone}_{model}_*"))
            if not worker_dirs:
                continue
            worker_dir = worker_dirs[-1]
            worker_csvs = sorted(worker_dir.glob("alpha_worker_*.csv"))
            if not worker_csvs:
                continue
            row = pd.read_csv(worker_csvs[0])
            bl = {}
            for col in ["base_jaccard_mean", "base_spearman_mean", "base_kendall_mean"]:
                if col in row.columns:
                    bl[col] = row[col].iloc[0]
            if bl:
                baselines[(model, clone)] = bl
    return baselines


# ---------------------------------------------------------------------------
# Summary table (per model x clone)
# ---------------------------------------------------------------------------

def compute_summary(combined: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """One row per model x clone_type with key statistics."""
    rows = []
    for (model, clone), grp in combined.groupby(["model", "clone_type"]):
        grp = grp.sort_values("alpha")

        best_idx = grp["crw_jaccard_mean"].idxmax()
        best = grp.loc[best_idx]
        peak_j = best["crw_jaccard_mean"]

        # Plateau: alphas where Jaccard >= 95% of peak
        threshold = peak_j * 0.95
        in_plateau = grp[grp["crw_jaccard_mean"] >= threshold]
        plateau_start = in_plateau["alpha"].min()
        plateau_end = in_plateau["alpha"].max()

        bl = baselines.get((model, clone), {})

        row = {
            "model": MODEL_DISPLAY.get(model, model),
            "clone_type": CLONE_DISPLAY.get(clone, clone),
            "baseline_jaccard": bl.get("base_jaccard_mean", np.nan),
            "baseline_spearman": bl.get("base_spearman_mean", np.nan),
            "baseline_kendall": bl.get("base_kendall_mean", np.nan),
            "optimal_alpha": best["alpha"],
            "peak_jaccard": peak_j,
            "peak_spearman": best["crw_spearman_mean"],
            "peak_kendall": best["crw_kendall_mean"],
            "jaccard_at_0.6": grp.loc[grp["alpha"] == 0.6, "crw_jaccard_mean"].values[0]
                if 0.6 in grp["alpha"].values else np.nan,
            "improvement": peak_j - bl.get("base_jaccard_mean", np.nan),
            "plateau_start": plateau_start,
            "plateau_end": plateau_end,
            "plateau_width": plateau_end - plateau_start,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    model_cat = pd.CategoricalDtype(
        [MODEL_DISPLAY[m] for m in MODEL_ORDER], ordered=True
    )
    clone_cat = pd.CategoricalDtype(
        [CLONE_DISPLAY[c] for c in CLONE_ORDER], ordered=True
    )
    summary["model"] = summary["model"].astype(model_cat)
    summary["clone_type"] = summary["clone_type"].astype(clone_cat)
    summary = summary.sort_values(["model", "clone_type"]).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Model summary (aggregated across clone types)
# ---------------------------------------------------------------------------

def compute_model_summary(
    combined: pd.DataFrame, summary: pd.DataFrame, baselines: dict,
) -> pd.DataFrame:
    """One row per model: aggregated metrics and optimal alpha range."""
    rows = []
    for model in MODEL_ORDER:
        display = MODEL_DISPLAY[model]
        model_data = combined[combined["model"] == model]
        model_summary = summary[summary["model"] == display]

        if model_data.empty:
            continue

        # Mean CRW Jaccard curve across clone types at each alpha
        mean_curve = model_data.groupby("alpha")["crw_jaccard_mean"].mean()
        mean_curve = mean_curve.sort_index()

        # Model optimal alpha (maximizes mean curve)
        model_optimal_alpha = mean_curve.idxmax()
        peak_mean_jaccard = mean_curve.max()

        # Model plateau: where mean curve >= 95% of peak
        threshold = peak_mean_jaccard * 0.95
        in_plateau = mean_curve[mean_curve >= threshold]
        model_plateau_start = in_plateau.index.min()
        model_plateau_end = in_plateau.index.max()
        model_plateau_width = model_plateau_end - model_plateau_start

        # Per-clone stats from summary
        avg_peak_jaccard = model_summary["peak_jaccard"].mean()
        avg_improvement = model_summary["improvement"].mean()
        worst_improvement = model_summary["improvement"].min()
        avg_clone_plateau_width = model_summary["plateau_width"].mean()

        # Model score = effectiveness x stability
        model_score = avg_improvement * model_plateau_width

        rows.append({
            "model": display,
            "avg_peak_jaccard": avg_peak_jaccard,
            "avg_improvement": avg_improvement,
            "worst_improvement": worst_improvement,
            "model_optimal_alpha": model_optimal_alpha,
            "model_plateau_start": model_plateau_start,
            "model_plateau_end": model_plateau_end,
            "model_plateau_width": model_plateau_width,
            "avg_clone_plateau_width": avg_clone_plateau_width,
            "model_score": model_score,
        })

    model_summary_df = pd.DataFrame(rows)
    model_summary_df = model_summary_df.sort_values(
        "model_score", ascending=False
    ).reset_index(drop=True)
    model_summary_df["rank"] = range(1, len(model_summary_df) + 1)
    return model_summary_df


def get_model_plateaus(model_summary_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Extract {model_key: (plateau_start, plateau_end)} for plotting."""
    # Map display names back to model keys
    display_to_key = {v: k for k, v in MODEL_DISPLAY.items()}
    plateaus = {}
    for _, row in model_summary_df.iterrows():
        key = display_to_key.get(row["model"], row["model"])
        plateaus[key] = (row["model_plateau_start"], row["model_plateau_end"])
    return plateaus


# ---------------------------------------------------------------------------
# Plot — Alpha curves per model (Jaccard or Spearman)
# ---------------------------------------------------------------------------

_METRIC_CONFIG = {
    "jaccard": {
        "col": "crw_jaccard_mean",
        "baseline_col": "base_jaccard_mean",
        "ylabel": "Jaccard Similarity (mean)",
        "title_metric": "Jaccard Similarity",
    },
    "spearman": {
        "col": "crw_spearman_mean",
        "baseline_col": "base_spearman_mean",
        "ylabel": "Spearman Correlation (mean)",
        "title_metric": "Spearman Correlation",
    },
}


def plot_alpha_curves(
    combined: pd.DataFrame,
    baselines: dict,
    model_plateaus: dict[str, tuple[float, float]],
    outpath: Path,
    metric: str = "jaccard",
):
    """2x2 facet: one subplot per model, lines per clone type, with optimal alpha band."""
    cfg = _METRIC_CONFIG[metric]
    col = cfg["col"]
    bl_col = cfg["baseline_col"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    axes_flat = axes.flat

    # Compute y-axis limits from data
    all_vals = combined[col].dropna()
    all_bl_vals = [
        bl[bl_col] for bl in baselines.values() if bl_col in bl
    ]
    y_min_data = min(all_vals.min(), min(all_bl_vals) if all_bl_vals else all_vals.min())
    y_max_data = max(all_vals.max(), max(all_bl_vals) if all_bl_vals else all_vals.max())
    y_padding = (y_max_data - y_min_data) * 0.08
    y_lo = y_min_data - y_padding
    y_hi = y_max_data + y_padding

    for ax, model in zip(axes_flat, MODEL_ORDER):
        model_data = combined[combined["model"] == model]

        # Shaded band for model optimal alpha range
        if model in model_plateaus:
            p_start, p_end = model_plateaus[model]
            ax.axvspan(
                p_start, p_end,
                color=_COLOR_BAND, alpha=0.10, zorder=1,
            )
            ax.axvline(p_start, color=_COLOR_BAND, linestyle=":", linewidth=1.0, alpha=0.5, zorder=2)
            ax.axvline(p_end, color=_COLOR_BAND, linestyle=":", linewidth=1.0, alpha=0.5, zorder=2)
            # Annotation
            ax.text(
                (p_start + p_end) / 2, y_hi - y_padding * 0.3,
                f"$\\alpha$ $\\in$ [{p_start:.1f}, {p_end:.1f}]",
                ha="center", va="top", fontsize=9, color=_COLOR_BAND,
                fontweight="bold", zorder=5,
            )

        for clone in CLONE_ORDER:
            subset = model_data[model_data["clone_type"] == clone].sort_values("alpha")
            if subset.empty:
                continue

            color = CLONE_COLORS[clone]
            ls = CLONE_LINESTYLES[clone]
            label = CLONE_DISPLAY[clone]

            ax.plot(
                subset["alpha"], subset[col],
                color=color, linestyle=ls, linewidth=1.8, label=label,
                marker="o", markersize=3, zorder=3,
            )

            # Baseline dashed line
            bl = baselines.get((model, clone), {})
            bl_val = bl.get(bl_col)
            if bl_val is not None:
                ax.axhline(bl_val, color=color, linestyle=":", linewidth=0.9, alpha=0.5)

            # Mark optimal alpha
            best_idx = subset[col].idxmax()
            best_alpha = subset.loc[best_idx, "alpha"]
            best_val = subset.loc[best_idx, col]
            ax.plot(best_alpha, best_val, marker="*", color=color, markersize=10, zorder=4)

        ax.set_title(MODEL_DISPLAY[model], fontsize=13, fontweight="bold")
        ax.set_ylim(y_lo, y_hi)

    for ax in axes[1, :]:
        ax.set_xlabel(r"Alpha ($\alpha$)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(cfg["ylabel"], fontsize=11)

    # Collect handles from all subplots (in case one is missing a clone type)
    all_handles = {}
    for ax_i in axes_flat:
        for h, l in zip(*ax_i.get_legend_handles_labels()):
            if l not in all_handles:
                all_handles[l] = h
    fig.legend(
        all_handles.values(), all_handles.keys(),
        loc="lower center", ncol=6, fontsize=10,
        bbox_to_anchor=(0.5, -0.02), frameon=True,
    )

    fig.suptitle(
        f"CRW Correction: {cfg['title_metric']} vs Alpha\n"
        f"(dotted = baseline, \u2605 = optimal \u03b1, "
        f"shaded band = usable \u03b1 range)",
        fontsize=14, y=0.98,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Model comparison heatmaps
# ---------------------------------------------------------------------------

def plot_model_heatmaps(summary: pd.DataFrame, outpath: Path):
    """Two heatmaps: peak CRW Jaccard and Jaccard improvement over baseline."""
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    metrics = [
        ("peak_jaccard", "Peak CRW Jaccard\n(best \u03b1 \u2014 higher = better correction)", "Greens"),
        ("improvement", "Jaccard Improvement over Baseline\n(Peak CRW \u2212 Baseline)", "RdYlGn"),
    ]

    for ax, (col, title, cmap) in zip(axes, metrics):
        pivot = summary.pivot(index="model", columns="clone_type", values=col)

        vmin = pivot.min().min()
        vmax = pivot.max().max()
        if col == "improvement":
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f", cmap=cmap,
            vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 11},
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

    fig.suptitle(
        "Model \u00d7 Clone-Type: CRW Effectiveness Overview",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Optimal alpha heatmap
# ---------------------------------------------------------------------------

def plot_optimal_alpha_heatmap(summary: pd.DataFrame, outpath: Path):
    """Heatmap of optimal alpha values (model x clone type)."""
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(9, 4.5))

    pivot = summary.pivot(index="model", columns="clone_type", values="optimal_alpha")

    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=1.5,
        cbar_kws={"label": "Optimal \u03b1", "shrink": 0.8},
    )
    ax.set_title("Optimal Alpha per Model \u00d7 Clone Type", fontsize=13, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)

    for i, model in enumerate(pivot.index):
        for j, clone in enumerate(pivot.columns):
            row = summary[
                (summary["model"] == model) & (summary["clone_type"] == clone)
            ]
            if not row.empty:
                pw = row["plateau_width"].values[0]
                ax.text(
                    j + 0.5, i + 0.75, f"\u00b1{pw:.1f}",
                    ha="center", va="center", fontsize=7, color="gray",
                )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Model ranking bar chart
# ---------------------------------------------------------------------------

def plot_model_ranking(model_summary_df: pd.DataFrame, outpath: Path):
    """Grouped bar chart comparing models on effectiveness and stability."""
    sns.set_theme(style="whitegrid")

    models = model_summary_df.sort_values("rank")["model"].tolist()
    n_models = len(models)

    metrics = [
        ("avg_improvement", "Avg Peak Improvement\n(Jaccard)"),
        ("model_plateau_width", "Usable \u03b1 Range\n(plateau width)"),
        ("worst_improvement", "Worst-case\nImprovement"),
    ]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    # Use a consistent color per model
    palette = sns.color_palette("Set2", n_models)
    model_colors = {m: palette[i] for i, m in enumerate(models)}

    for ax, (col, label) in zip(axes, metrics):
        vals = [model_summary_df.loc[model_summary_df["model"] == m, col].values[0] for m in models]
        bars = ax.bar(range(n_models), vals, color=[model_colors[m] for m in models])

        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 0.1)

    # Add rank annotation
    fig.suptitle(
        "Model Ranking: CRW Effectiveness \u00d7 Stability\n"
        "(ranked by score = avg improvement \u00d7 plateau width)",
        fontsize=13, y=1.02,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Plot — Model summary table
# ---------------------------------------------------------------------------

def plot_model_table(model_summary_df: pd.DataFrame, outpath: Path):
    """Render model summary as a publication-quality table figure."""
    sns.set_theme(style="white")

    df = model_summary_df.sort_values("rank").copy()

    # Build table data
    col_labels = [
        "Rank", "Model", "Optimal \u03b1",
        "Usable \u03b1 Range", "Plateau\nWidth",
        "Avg Peak\nJaccard", "Avg\nImprovement",
        "Worst-case\nImprovement", "Score",
    ]
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([
            f"{row['rank']:.0f}",
            row["model"],
            f"{row['model_optimal_alpha']:.1f}",
            f"[{row['model_plateau_start']:.1f}, {row['model_plateau_end']:.1f}]",
            f"{row['model_plateau_width']:.1f}",
            f"{row['avg_peak_jaccard']:.3f}",
            f"{row['avg_improvement']:.3f}",
            f"{row['worst_improvement']:.3f}",
            f"{row['model_score']:.4f}",
        ])

    n_rows = len(cell_text)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.55 * n_rows))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_height(0.18)

    # Style data rows: alternate shading, highlight rank 1
    palette = ["#FAFAFA", "#ECEFF1"]
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(palette[i % 2])
            cell.set_text_props(fontsize=10)
            cell.set_height(0.14)
            # Bold the top-ranked row
            if i == 0:
                cell.set_text_props(fontweight="bold", fontsize=10)
                cell.set_facecolor("#E3F2FD")

    table.auto_set_column_width(list(range(n_cols)))

    ax.set_title(
        "Model Summary: CRW Effectiveness and Optimal \u03b1 Range\n"
        "(score = avg improvement \u00d7 plateau width; "
        "plateau = \u03b1 range where mean Jaccard \u2265 95% of peak)",
        fontsize=12, pad=15, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading alpha sweep CSVs (excluding identical clones)...")
    combined = load_main_csvs()
    n_combos = combined.groupby(["model", "clone_type"]).ngroups
    print(f"  Found {len(combined)} rows across {n_combos} model x clone combinations")

    print("Loading baseline metrics from worker CSVs...")
    baselines = load_baseline_metrics()
    print(f"  Found baselines for {len(baselines)} combinations")

    # --- Per model x clone summary ---
    print("Computing per model x clone summary...")
    summary = compute_summary(combined, baselines)
    summary_path = OUTPUT_DIR / "alpha_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  -> {summary_path.name}")

    print("\n" + "=" * 80)
    print("SUMMARY: Optimal alpha and peak Jaccard per model x clone type")
    print("=" * 80)
    display_cols = [
        "model", "clone_type", "baseline_jaccard", "peak_jaccard",
        "improvement", "optimal_alpha", "plateau_width",
    ]
    print(summary[display_cols].to_string(index=False, float_format="%.3f"))

    # --- Model-level summary ---
    print("\nComputing model-level summary...")
    model_summary_df = compute_model_summary(combined, summary, baselines)
    model_summary_path = OUTPUT_DIR / "alpha_sweep_model_summary.csv"
    model_summary_df.to_csv(model_summary_path, index=False)
    print(f"  -> {model_summary_path.name}")

    print("\n" + "=" * 80)
    print("MODEL RANKING (by score = avg improvement x plateau width)")
    print("=" * 80)
    model_display_cols = [
        "rank", "model", "avg_peak_jaccard", "avg_improvement",
        "worst_improvement", "model_optimal_alpha",
        "model_plateau_start", "model_plateau_end", "model_plateau_width",
        "model_score",
    ]
    print(model_summary_df[model_display_cols].to_string(index=False, float_format="%.3f"))

    # --- Plots ---
    model_plateaus = get_model_plateaus(model_summary_df)

    print("\nGenerating plots...")
    plot_alpha_curves(
        combined, baselines, model_plateaus,
        OUTPUT_DIR / "alpha_curves_jaccard.png", metric="jaccard",
    )
    plot_alpha_curves(
        combined, baselines, model_plateaus,
        OUTPUT_DIR / "alpha_curves_spearman.png", metric="spearman",
    )
    plot_model_heatmaps(summary, OUTPUT_DIR / "model_comparison_heatmaps.png")
    plot_optimal_alpha_heatmap(summary, OUTPUT_DIR / "optimal_alpha_heatmap.png")
    plot_model_ranking(model_summary_df, OUTPUT_DIR / "model_ranking.png")
    plot_model_table(model_summary_df, OUTPUT_DIR / "model_summary_table.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
