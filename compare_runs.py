import argparse
from configs import base_constants as config
from cross_run_analysis.analyzer import CrossRunAnalyzer
from cross_run_analysis.saver import CrossRunSaver


def main():
    parser = argparse.ArgumentParser(description="Compare two VAA recommendation runs.")
    parser.add_argument("run_a", type=str, help="Path to first parquet file")
    parser.add_argument("run_b", type=str, help="Path to second parquet file")
    parser.add_argument(
        "-n", "--n_value", type=int, default=None, help="Override Jaccard N"
    )

    args = parser.parse_args()

    # 1. Analyze
    analyzer = CrossRunAnalyzer(args.run_a, args.run_b, n_override=args.n_value)
    print(
        f"Analyzer loaded with:\n  Run A: {args.run_a}\n  Run B: {args.run_b}\n  N override: {args.n_value}"
    )
    results_df = analyzer.analyze()

    # 2. Save
    print(f"\nSaving results...")
    saver = CrossRunSaver(output_dir=config.COMPARISON_RESULTS_DIR)  # type: ignore

    saver.save_results(
        df=results_df,
        meta_a=analyzer.meta_a,
        meta_b=analyzer.meta_b,
        n_used=analyzer.n,
    )


if __name__ == "__main__":
    main()
