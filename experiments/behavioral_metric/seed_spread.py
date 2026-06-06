"""
Per-seed spread of the held-out distortion, from a deployment `…_per_seed.csv`.

Shows, for each α, the individual value from each 5% pilot (seed) — so the across-seed spread is
visible directly rather than hidden inside an aggregate band. Three auto-scaled panels (Jaccard /
Spearman / Kendall distortion) so even the tiny-magnitude metrics reveal their seed spread.

Note: per-seed *means* are averaged over ~9k held-out voters, so they are inherently tight (this
plot makes that visible and honest); the stronger inter-pilot test is per-voter recommendation
agreement, reported as `rec_stability_*` in the deployment.

Usage:
    python -m main behavioral-seed-spread \
        --per-seed-csv experiment_results/behavioral_metric/deployment_sim/deployment_..._per_seed.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRICS = [("jaccard_mean", "Jaccard"), ("spearman_mean", "Spearman"), ("kendall_mean", "Kendall")]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-seed distortion spread from a deployment per_seed CSV")
    p.add_argument("--per-seed-csv", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None, help="Default: alongside the input CSV")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    csv_path = Path(args.per_seed_csv)
    df = pd.read_csv(csv_path)
    seeds = sorted(df["seed"].unique())
    colors = plt.cm.tab10(range(len(seeds)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for ax, (col, name) in zip(axes, METRICS):
        for seed, color in zip(seeds, colors):
            sub = df[df["seed"] == seed].sort_values("alpha")
            ax.plot(sub["alpha"], 1 - sub[col], marker="o", ms=4, lw=0.8,
                    color=color, alpha=0.85, label=f"seed {seed}")
        mean = df.groupby("alpha")[col].mean()
        ax.plot(mean.index, 1 - mean.values, color="black", lw=2.2, label="mean", zorder=5)
        ax.set_ylabel(f"{name} distortion\n(1 − agreement)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    axes[0].set_title("Per-seed held-out distortion vs α — each line = one 5% pilot\n"
                      "(axes auto-scaled per metric so the across-seed spread is visible)")
    axes[-1].set_xlabel("Alpha (α)")
    fig.tight_layout()

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (csv_path.stem.replace("_per_seed", "") + "_seed_spread.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
