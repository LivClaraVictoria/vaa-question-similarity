import argparse
from cross_run_analysis.analyzer import CrossRunAnalyzer
from cross_run_analysis.saver import CrossRunSaver


"""
PARSER FORMAT
Standard run (Auto-detects settings from JSON):
python run_comparison.py experiment_results/runA.parquet experiment_results/runB.parquet

Override run (Forces Jaccard n=50):
python run_comparison.py experiment_results/runA.parquet experiment_results/runB.parquet -n 50
"""


def main():
    parser = argparse.ArgumentParser(description="Compare two VAA recommendation runs.")

    # Required arguments: The two parquet files
    parser.add_argument(
        "run_a", type=str, help="Path to the first parquet file (standard data)"
    )
    parser.add_argument(
        "run_b", type=str, help="Path to the second parquet file (cloned data)"
    )

    # Optional override: Force a specific 'n' for Jaccard
    parser.add_argument(
        "-n",
        "--n_value",
        type=int,
        default=None,
        help="Override the Jaccard 'n' value (ignores metadata)",
    )

    args = parser.parse_args()

    print(f"--- VAA Cross-Run Comparator ---")
    print(f"Run A: {args.run_a}")
    print(f"Run B: {args.run_b}")

    try:
        # 1. Initialize and Run Analysis
        # The analyzer handles loading, config comparison, and math
        analyzer = CrossRunAnalyzer(args.run_a, args.run_b, n_override=args.n_value)
        results_df = analyzer.analyze()

        # 2. Save Results
        # The saver handles hashing, text summary generation, and file I/O
        saver = CrossRunSaver()
        saver.save_results(
            df=results_df,
            meta_a=analyzer.meta_a,
            meta_b=analyzer.meta_b,
            n_used=analyzer.n,
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Optional: Print full traceback for debugging if needed
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    main()
