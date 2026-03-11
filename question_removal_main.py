"""
Create a reduced dataset by removing questions from a specific category.

Writes the reduced dataset to data/removed/{clone_id}/ following the same
file conventions as cloned datasets, so the existing pipeline and alpha_sweep_main.py
can process it without modification.

Also auto-generates a pipeline config for the reduced dataset.

Usage:
    python -m question_removal_main \
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py \
        --category "Health" \
        --n-remove 3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from main import load_config
from question_removal.remover import get_category_questions, remove_questions
from vqs.data_loader import load_dataset

# Path to question impact CSV (for sorting by impact)
IMPACT_DIR = Path("experiment_results/question_impact")


def _find_impact_csv() -> Path | None:
    csvs = sorted(IMPACT_DIR.glob("question_impact_*.csv"))
    return csvs[-1] if csvs else None


def _sort_by_impact(question_ids: list[int], ascending: bool = True) -> list[int]:
    """Sort question IDs by impact (lowest first by default). Falls back to ID order."""
    impact_path = _find_impact_csv()
    if impact_path is None:
        print("  No impact CSV found — sorting by question ID instead")
        return sorted(question_ids)

    impact_df = pd.read_csv(impact_path)
    impact_lookup = impact_df.set_index("question_id")["impact"].to_dict()

    # Partition into found/not-found
    found = [(qid, impact_lookup[qid]) for qid in question_ids if qid in impact_lookup]
    not_found = [qid for qid in question_ids if qid not in impact_lookup]

    if not_found:
        print(f"  Warning: {len(not_found)} questions not in impact CSV: {not_found}")

    found.sort(key=lambda x: x[1], reverse=not ascending)
    result = [qid for qid, _ in found] + sorted(not_found)
    return result


def _find_file_by_prefix(directory: Path, prefix: str) -> Path:
    """Find a single file matching prefix*.parquet in directory."""
    files = list(directory.glob(f"{prefix}*.parquet"))
    if len(files) == 0:
        raise FileNotFoundError(f"No file found with prefix '{prefix}' in {directory}")
    return files[0]


def _write_reduced_dataset(
    dataframes: dict[str, pd.DataFrame],
    out_dir: Path,
    source_dir: Path,
    category: str,
    removed_ids: list[int],
    retained_ids: list[int],
    data_year: int,
) -> None:
    """Write the reduced dataset to disk, following cloned data conventions."""
    if out_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {out_dir}\n"
            f"Delete it manually if you want to regenerate."
        )
    out_dir.mkdir(parents=True)

    # Write parquets — use the same filenames as source for voters/candidates
    # (data_loader uses prefix matching, so the hash suffix doesn't matter)
    questions_path = out_dir / "df_questions.parquet"
    dataframes["questions"].to_parquet(questions_path)

    # Find source filenames to reuse the naming convention
    voters_src = _find_file_by_prefix(source_dir, "df_voters-")
    candidates_src = _find_file_by_prefix(source_dir, "df_candidates-")

    dataframes["voters"].to_parquet(out_dir / voters_src.name)
    dataframes["candidates"].to_parquet(out_dir / candidates_src.name)

    # Write metadata
    metadata = {
        "data_year": data_year,
        "source_dir": str(source_dir),
        "category": category,
        "removed_question_ids": removed_ids,
        "retained_question_ids": retained_ids,
        "n_removed": len(removed_ids),
        "n_retained": len(retained_ids),
        "n_total_original": len(removed_ids) + len(retained_ids),
        "generated_at": datetime.now().isoformat(),
    }
    with open(out_dir / "removal_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  -> {questions_path}")
    print(f"  -> {out_dir / voters_src.name}")
    print(f"  -> {out_dir / candidates_src.name}")
    print(f"  -> {out_dir / 'removal_metadata.json'}")


def _generate_pipeline_config(
    config,
    clone_id: str,
    config_dir: Path,
    config_name: str,
) -> Path:
    """Auto-generate a pipeline config for the reduced dataset."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_name

    # Extract model-specific fields from the base config
    dist = getattr(config, "dist", "E5-INSTRUCT")
    embedding_instruction = getattr(config, "embedding_instruction", "")
    embedding_task = getattr(config, "embedding_task", "")
    district = getattr(config, "district", "all")
    n_recommendations = getattr(config, "n_recommendations", "all")

    lines = [
        f'# Auto-generated config for reduced dataset: {clone_id}',
        f'# Generated: {datetime.now().isoformat()}',
        f'from configs.full_pipeline.cloned.base_cloned import *',
        f'',
        f'clone_id = "{clone_id}"',
        f'',
        f'CLEANED_DIR = REMOVED_DIR / clone_id',
        f'QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"',
        f'QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"',
        f'RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"',
        f'',
        f'dist = "{dist}"',
    ]

    if embedding_instruction:
        lines.append(f'embedding_instruction = "{embedding_instruction}"')
    if embedding_task:
        lines.append(f'embedding_task = "{embedding_task}"')

    lines.append(f'district = "{district}"')
    lines.append(f'')
    lines.append(f'n_recommendations = "{n_recommendations}"')
    lines.append(f'')

    config_path.write_text("\n".join(lines))
    return config_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reduced dataset by removing questions from a category"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Category name to remove questions from (e.g. 'Health'). Required unless --remove-ids is used.",
    )
    parser.add_argument(
        "--n-remove", type=int, default=None,
        help="Number of questions to remove from category",
    )
    parser.add_argument(
        "--remove-ids", type=str, default=None,
        help="Explicit comma-separated question IDs to remove (skips category requirement)",
    )
    parser.add_argument(
        "--highest", action="store_true",
        help="Remove highest-impact questions first (default: lowest-impact first)",
    )
    parser.add_argument(
        "--clone-id", type=str, default=None,
        help="Override auto-generated clone_id",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Load config
    config = load_config(Path(args.config))

    print(f"\n=== Question Removal: Create Reduced Dataset ===")
    print(f"  Base config: {args.config}")

    # Load questions to find category members
    questions_path = config.QUESTIONS_2023_PATH
    df_questions = pd.read_parquet(questions_path)
    print(f"  Questions:   {len(df_questions)} total")

    # Determine which questions to remove
    if args.remove_ids:
        # Explicit IDs mode — no category required
        remove_ids = [int(x.strip()) for x in args.remove_ids.split(",")]
        all_qids = set(df_questions["ID_question"].tolist())
        invalid = [qid for qid in remove_ids if qid not in all_qids]
        if invalid:
            print(f"ERROR: IDs {invalid} not found in questions DataFrame", file=sys.stderr)
            sys.exit(1)
        retained_ids = [qid for qid in all_qids if qid not in set(remove_ids)]
        print(f"  Removing {len(remove_ids)} explicit IDs: {remove_ids}")
    else:
        # Category mode
        if not args.category:
            print("ERROR: --category is required unless --remove-ids is used", file=sys.stderr)
            sys.exit(1)
        if not args.n_remove:
            print("ERROR: --n-remove is required unless --remove-ids is used", file=sys.stderr)
            sys.exit(1)

        print(f"  Category:    {args.category}")
        category_qids = get_category_questions(df_questions, args.category)
        print(f"  Category '{args.category}': {len(category_qids)} questions — {category_qids}")

        n_remove = args.n_remove
        if n_remove >= len(category_qids):
            print(
                f"ERROR: Cannot remove {n_remove} questions from category with {len(category_qids)} questions",
                file=sys.stderr,
            )
            sys.exit(1)

        # Sort by impact: ascending=True means lowest-first (default), highest flag reverses
        ascending = not args.highest
        sorted_qids = _sort_by_impact(category_qids, ascending=ascending)
        remove_ids = sorted_qids[:n_remove]
        retained_ids = [qid for qid in category_qids if qid not in remove_ids]

    print(f"\n  Removing {len(remove_ids)} questions: {remove_ids}")
    print(f"  Retaining {len(retained_ids)} in category")

    # Build clone_id
    if args.clone_id:
        clone_id = args.clone_id
    elif args.category:
        category_slug = args.category.lower().replace(" & ", "_").replace(" ", "_")
        n_cat = len(get_category_questions(df_questions, args.category))
        clone_id = f"removed_{category_slug}_{len(remove_ids)}of{n_cat}"
    else:
        clone_id = f"removed_{len(remove_ids)}q"
    print(f"  clone_id:    {clone_id}")

    # Load full dataset
    print(f"\n--- Loading full dataset ---")
    dataset = load_dataset(config)

    # Remove questions
    print(f"\n--- Removing questions ---")
    reduced = remove_questions(dataset, remove_ids)
    n_orig = len(dataset["questions"])
    n_reduced = len(reduced["questions"])
    print(f"  Questions: {n_orig} -> {n_reduced}")

    # Verify column removal
    for qid in remove_ids:
        for df_name in ["voters", "candidates"]:
            assert f"answer_{qid}" not in reduced[df_name].columns, \
                f"answer_{qid} still in {df_name}!"
        assert f"weight_{qid}" not in reduced["voters"].columns, \
            f"weight_{qid} still in voters!"
    print(f"  Column verification passed")

    # Write to disk
    out_dir = config.REMOVED_DIR / clone_id
    print(f"\n--- Writing reduced dataset to {out_dir} ---")
    _write_reduced_dataset(
        dataframes=reduced,
        out_dir=out_dir,
        source_dir=config.CLEANED_DIR,
        category=args.category,
        removed_ids=remove_ids,
        retained_ids=retained_ids,
        data_year=config.data_year,
    )

    # Auto-generate pipeline config
    config_name = f"{clone_id}_{config.dist.lower().replace('-', '_')}_{config.district}.py"
    config_dir = Path("configs/full_pipeline/removed")
    print(f"\n--- Generating pipeline config ---")
    config_path = _generate_pipeline_config(
        config=config,
        clone_id=clone_id,
        config_dir=config_dir,
        config_name=config_name,
    )
    print(f"  -> {config_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUCCESS: Reduced dataset created")
    print(f"  Data:   {out_dir}")
    print(f"  Config: {config_path}")
    print(f"")
    print(f"To run the alpha sweep (Part 2):")
    print(f"  python -m alpha_sweep_main \\")
    print(f"      --config_a {args.config} \\")
    print(f"      --config_b {config_path} \\")
    print(f"      --output-dir experiment_results/exp2_question_removal")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
