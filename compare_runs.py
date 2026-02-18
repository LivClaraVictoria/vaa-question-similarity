import argparse
from configs import base_constants as config
from cross_run_analysis.analyzer import CrossRunAnalyzer
from cross_run_analysis.saver import CrossRunSaver
from cross_run_analysis.computation_cache import ComputationCache


def main():
    parser = argparse.ArgumentParser(description="Compare two VAA recommendation runs.")
    parser.add_argument(
        "run_a", type=str, help="Path to first run parquet file (base dataset)"
    )
    parser.add_argument(
        "run_b", type=str, help="Path to second run parquet file (cloned dataset)"
    )
    parser.add_argument(
        "-n",
        "--n_value",
        type=int,
        default=None,
        help="Override Jaccard top-k (default: min of n_jaccard from each run's metadata)",
    )
    args = parser.parse_args()

    cache = ComputationCache(config.COMPARATOR_CACHE_DIR)
    analyzer = CrossRunAnalyzer(args.run_a, args.run_b, n_override=args.n_value)
    results_df = analyzer.analyze(cache=cache)

    saver = CrossRunSaver(output_dir=config.COMPARISON_RESULTS_DIR)
    saver.save_results(
        df=results_df,
        meta_a=analyzer.meta_a,
        meta_b=analyzer.meta_b,
        n_used=analyzer.n,
    )


if __name__ == "__main__":
    main()
